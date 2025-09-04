# demonology/tools/project.py
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool

logger = logging.getLogger(__name__)


class ProjectPlanningTool(Tool):
    """Analyze existing projects or generate new project plans and task breakdowns for development projects."""
    
    def __init__(self):
        super().__init__("project_planning", "Analyze existing projects and continue work, or generate new project plans and task breakdowns")
    
    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["analyze_existing", "create_new", "continue"], "description": "Action to perform: analyze existing project, create new project, or continue existing work", "default": "analyze_existing"},
                    "project_name": {"type": "string", "description": "Name of the project (for new projects) or path to existing project directory"},
                    "project_description": {"type": "string", "description": "Description of what the project should do (for new projects)"},
                    "technology_stack": {"type": "string", "description": "Preferred technologies (e.g., Python, React, C++)"},
                    "complexity": {"type": "string", "enum": ["simple", "medium", "complex"], "description": "Project complexity level"},
                    "save_to_file": {"type": "boolean", "description": "Whether to save the plan to a file", "default": True},
                    "execute_plan": {"type": "boolean", "description": "Whether to automatically create the project structure", "default": True}
                },
                "required": ["action", "project_name"]
            }
        }
    
    async def execute(self, action: str = "analyze_existing", project_name: str = "", project_description: str = "", 
                     technology_stack: str = "", complexity: str = "medium", save_to_file: bool = True, 
                     execute_plan: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            if action == "analyze_existing":
                return await self._analyze_existing_project(project_name, save_to_file)
            elif action == "continue":
                return await self._continue_existing_project(project_name, save_to_file)
            elif action == "create_new":
                return await self._create_new_project(project_name, project_description, technology_stack, 
                                                    complexity, save_to_file, execute_plan)
            else:
                # Default behavior - try to analyze existing first, then create new
                existing_analysis = await self._analyze_existing_project(project_name, False)
                if existing_analysis["success"] and existing_analysis.get("project_found", False):
                    return existing_analysis
                else:
                    return await self._create_new_project(project_name, project_description, technology_stack, 
                                                        complexity, save_to_file, execute_plan)
                
        except Exception as e:
            logger.exception("ProjectPlanningTool error")
            return {"success": False, "error": str(e)}
    
    async def _analyze_existing_project(self, project_name: str, save_to_file: bool = True) -> Dict[str, Any]:
        """Analyze an existing project and provide continuation suggestions."""
        try:
            # Look for existing project directory
            project_path = None
            
            # Try different approaches to find the project
            if "/" in project_name or "\\" in project_name:
                # Path provided - check if it's a file or directory
                try:
                    input_path = Path(project_name).resolve()
                    
                    # If it's a file, use its parent directory
                    if input_path.exists() and input_path.is_file():
                        project_path = input_path.parent
                        # If it's a plan file, try to infer the project directory name
                        if input_path.name.endswith('_plan.md'):
                            # Look for directory with similar name
                            inferred_name = input_path.name.replace('_plan.md', '')
                            inferred_dir = input_path.parent / inferred_name
                            if inferred_dir.exists() and inferred_dir.is_dir():
                                project_path = inferred_dir
                    # If it's a directory, use it directly
                    elif input_path.exists() and input_path.is_dir():
                        project_path = input_path
                    else:
                        project_path = None
                        
                except (PermissionError, ValueError, OSError):
                    project_path = None
            else:
                # Try to find project in current directory or subdirectories
                try:
                    search_dir = Path.cwd().resolve()
                except Exception:
                    search_dir = Path.cwd().resolve()
                
                # Look for exact match
                potential_path = search_dir / project_name.replace(' ', '_').lower()
                if potential_path.exists() and potential_path.is_dir():
                    project_path = potential_path
                else:
                    # Look for similar names
                    for item in search_dir.iterdir():
                        if item.is_dir() and project_name.lower().replace(' ', '_') in item.name.lower():
                            project_path = item
                            break
            
            if not project_path or not project_path.exists():
                try:
                    current_dir = Path.cwd().resolve()
                    search_locations = [str(current_dir)]
                except Exception:
                    search_locations = [str(Path.cwd().resolve())]
                    
                return {
                    "success": True,
                    "project_found": False,
                    "message": f"No existing project found for '{project_name}'. Consider creating a new project.",
                    "search_locations": search_locations
                }
            
            # Analyze the existing project
            analysis = await self._perform_project_analysis(project_path)
            
            # Generate continuation plan
            continuation_plan = await self._generate_continuation_plan(project_path, analysis)
            
            result = {
                "success": True,
                "project_found": True,
                "project_path": str(project_path),
                "project_name": project_path.name,
                "analysis": analysis,
                "continuation_plan": continuation_plan,
                "message": f"Analyzed existing project '{project_path.name}' at {project_path}"
            }
            
            if save_to_file:
                # Save continuation plan
                plan_filename = f"{project_path.name}_continuation_plan.md"
                plan_file = project_path / plan_filename
                plan_file.write_text(continuation_plan, encoding="utf-8")
                result["plan_file"] = str(plan_file)
            
            return result
            
        except Exception as e:
            logger.exception("Project analysis error")
            return {"success": False, "error": str(e)}
    
    async def _continue_existing_project(self, project_name: str, save_to_file: bool = True) -> Dict[str, Any]:
        """Continue work on an existing project."""
        analysis_result = await self._analyze_existing_project(project_name, False)
        
        if not analysis_result["success"] or not analysis_result.get("project_found", False):
            return analysis_result
        
        project_path = Path(analysis_result["project_path"])
        analysis = analysis_result["analysis"]
        
        # Generate specific next steps
        next_steps = await self._generate_next_steps(project_path, analysis)
        
        result = {
            "success": True,
            "project_path": str(project_path),
            "project_name": project_path.name,
            "analysis": analysis,
            "next_steps": next_steps,
            "message": f"Ready to continue work on '{project_path.name}'"
        }
        
        if save_to_file:
            # Save next steps
            steps_filename = f"{project_path.name}_next_steps.md"
            steps_file = project_path / steps_filename
            steps_file.write_text(next_steps, encoding="utf-8")
            result["steps_file"] = str(steps_file)
        
        return result
    
    async def _create_new_project(self, project_name: str, project_description: str, technology_stack: str, 
                                complexity: str, save_to_file: bool, execute_plan: bool) -> Dict[str, Any]:
        """Create a new project (original functionality)."""
        # Generate project structure based on complexity and technology
        plan = self._generate_project_plan(project_name, project_description, technology_stack, complexity)
        
        result = {
            "success": True,
            "project_name": project_name,
            "plan": plan,
            "file_saved": None,
            "project_created": False,
            "files_created": []
        }
        
        if save_to_file:
            # Save to project plan file in the current working directory
            plan_filename = f"{project_name.replace(' ', '_').lower()}_plan.md"
            
            # Always save in current working directory
            try:
                current_dir = Path.cwd().resolve()
                plan_file = current_dir / plan_filename
            except Exception:
                # If we can't get current directory, use home directory as fallback
                plan_file = Path.home() / plan_filename
            
            plan_file.write_text(plan, encoding="utf-8")
            result["file_saved"] = str(plan_file)
        
        if execute_plan:
            # Create the actual project structure
            created_files = await self._execute_project_plan(project_name, technology_stack, complexity)
            result["project_created"] = True
            result["files_created"] = created_files
            result["message"] = f"Project '{project_name}' created with {len(created_files)} files"
        else:
            result["message"] = f"Project plan generated for '{project_name}'"
        
        return result
    
    async def _perform_project_analysis(self, project_path: Path) -> Dict[str, Any]:
        """Analyze an existing project's structure and contents."""
        analysis = {
            "project_type": "unknown",
            "files_found": [],
            "key_files": {},
            "dependencies": [],
            "structure": {},
            "last_modified": None,
            "size_info": {}
        }
        
        try:
            # Get basic info
            if project_path.exists():
                analysis["last_modified"] = project_path.stat().st_mtime
                
            # Analyze file structure
            total_files = 0
            total_size = 0
            file_types = {}
            
            for item in project_path.rglob("*"):
                if item.is_file():
                    total_files += 1
                    try:
                        size = item.stat().st_size
                        total_size += size
                        ext = item.suffix.lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
                    except:
                        pass
                    
                    # Record first 50 files
                    if len(analysis["files_found"]) < 50:
                        analysis["files_found"].append(str(item.relative_to(project_path)))
            
            analysis["size_info"] = {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "file_types": file_types
            }
            
            # Detect project type and key files
            key_files = {
                "package.json": project_path / "package.json",
                "requirements.txt": project_path / "requirements.txt", 
                "Cargo.toml": project_path / "Cargo.toml",
                "Makefile": project_path / "Makefile",
                "CMakeLists.txt": project_path / "CMakeLists.txt",
                "setup.py": project_path / "setup.py",
                "pyproject.toml": project_path / "pyproject.toml",
                "README.md": project_path / "README.md",
                "README.txt": project_path / "README.txt",
                "main.py": project_path / "main.py",
                "src/main.py": project_path / "src" / "main.py",
                "main.cpp": project_path / "main.cpp",
                "src/main.cpp": project_path / "src" / "main.cpp"
            }
            
            for name, path in key_files.items():
                if path.exists():
                    analysis["key_files"][name] = str(path.relative_to(project_path))
                    
            # Determine project type
            if "package.json" in analysis["key_files"]:
                analysis["project_type"] = "nodejs/javascript"
                # Try to read package.json for dependencies
                try:
                    pkg_json = json.loads((project_path / "package.json").read_text())
                    deps = list(pkg_json.get("dependencies", {}).keys())
                    dev_deps = list(pkg_json.get("devDependencies", {}).keys())
                    analysis["dependencies"] = deps + dev_deps
                except:
                    pass
            elif "requirements.txt" in analysis["key_files"] or "setup.py" in analysis["key_files"]:
                analysis["project_type"] = "python"
                # Try to read requirements
                try:
                    if "requirements.txt" in analysis["key_files"]:
                        req_content = (project_path / "requirements.txt").read_text()
                        analysis["dependencies"] = [line.strip() for line in req_content.split('\n') if line.strip() and not line.startswith('#')]
                except:
                    pass
            elif "Cargo.toml" in analysis["key_files"]:
                analysis["project_type"] = "rust"
            elif "CMakeLists.txt" in analysis["key_files"] or "Makefile" in analysis["key_files"]:
                analysis["project_type"] = "cpp/c"
            
        except Exception as e:
            logger.warning(f"Error analyzing project {project_path}: {e}")
        
        return analysis
    
    async def _generate_continuation_plan(self, project_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate a plan for continuing work on an existing project."""
        project_name = project_path.name
        project_type = analysis.get("project_type", "unknown")
        key_files = analysis.get("key_files", {})
        
        plan = f"""# {project_name} - Project Continuation Analysis

## Project Overview
- **Location**: {project_path}
- **Type**: {project_type}
- **Total Files**: {analysis.get('size_info', {}).get('total_files', 'Unknown')}
- **Total Size**: {analysis.get('size_info', {}).get('total_size_bytes', 0)} bytes

## Project Structure Analysis
"""
        
        if key_files:
            plan += "\n### Key Files Found:\n"
            for file_name, file_path in key_files.items():
                plan += f"- **{file_name}**: `{file_path}`\n"
        
        if analysis.get("dependencies"):
            plan += f"\n### Dependencies ({len(analysis['dependencies'])}):\n"
            for dep in analysis["dependencies"][:10]:  # Show first 10
                plan += f"- {dep}\n"
            if len(analysis["dependencies"]) > 10:
                plan += f"- ... and {len(analysis['dependencies']) - 10} more\n"
        
        plan += f"""
## Recommended Next Steps

Based on the project analysis, here are suggested actions to continue development:

### 1. Environment Setup
"""
        
        if project_type == "python":
            plan += """- Set up virtual environment: `python -m venv venv`
- Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`
"""
        elif project_type == "nodejs/javascript":
            plan += """- Install dependencies: `npm install`
- Check available scripts: `npm run`
"""
        elif project_type == "rust":
            plan += """- Build project: `cargo build`
- Run tests: `cargo test`
"""
        elif project_type == "cpp/c":
            plan += """- Build project: `make` or `cmake .` then `make`
"""
        
        plan += """
### 2. Code Review
- Review recent changes and TODOs
- Check for incomplete features or bug fixes
- Review documentation for outdated information

### 3. Testing & Quality
- Run existing tests to ensure current state is stable
- Check for linting or formatting issues
- Review error logs if any

### 4. Development Priorities
Based on the project structure, consider:
- Completing any incomplete features
- Adding missing documentation
- Improving test coverage
- Optimizing performance bottlenecks
- Adding new features as needed

---
*Analysis generated by Demonology Project Planning Tool*
"""
        
        return plan
    
    async def _generate_next_steps(self, project_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate specific actionable next steps for continuing the project."""
        project_name = project_path.name
        project_type = analysis.get("project_type", "unknown")
        
        steps = f"""# Next Steps for {project_name}

## Immediate Actions

### 1. Project Setup
"""
        
        if project_type == "python":
            steps += """```bash
cd """ + str(project_path) + """
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
python -m pytest  # Run tests if available
```
"""
        elif project_type == "nodejs/javascript":
            steps += """```bash
cd """ + str(project_path) + """
npm install
npm test  # Run tests if available
npm start  # Start development server
```
"""
        elif project_type == "rust":
            steps += """```bash
cd """ + str(project_path) + """
cargo build
cargo test
cargo run
```
"""
        
        steps += """
### 2. Code Analysis
- [ ] Read through main entry points and core modules
- [ ] Identify any TODO comments or incomplete features
- [ ] Check for any compilation or runtime errors
- [ ] Review recent git history (if available)

### 3. Development Focus Areas
- [ ] Complete any unfinished features
- [ ] Fix any identified bugs or issues  
- [ ] Add missing error handling
- [ ] Improve documentation where needed
- [ ] Add or improve tests for critical functionality

### 4. Quality Improvements
- [ ] Run linting tools and fix style issues
- [ ] Check for security vulnerabilities
- [ ] Optimize performance if needed
- [ ] Update dependencies if outdated

## Development Workflow
1. Make small, incremental changes
2. Test frequently during development
3. Commit changes regularly with clear messages
4. Document new features or significant changes
5. Consider adding integration tests for new functionality

---
*Generated by Demonology Project Planning Tool*
"""
        
        return steps
    
    def _generate_project_plan(self, name: str, description: str, tech_stack: str, complexity: str) -> str:
        """Generate a detailed project plan."""
        
        # Determine project phases based on complexity
        phases = {
            "simple": ["Planning", "Implementation", "Testing"],
            "medium": ["Planning", "Design", "Implementation", "Testing", "Documentation"],
            "complex": ["Planning", "Research", "Design", "Implementation", "Testing", "Documentation", "Deployment"]
        }
        
        # Generate file structure suggestions based on technology
        file_structure = self._suggest_file_structure(tech_stack.lower(), complexity)
        
        plan = f"""# {name} - Project Plan

## Project Overview
{description}

**Technology Stack:** {tech_stack or 'Not specified'}
**Complexity Level:** {complexity.title()}

## Project Phases

"""
        
        for i, phase in enumerate(phases.get(complexity, phases["medium"]), 1):
            plan += f"### {i}. {phase}\n"
            plan += self._get_phase_details(phase, complexity) + "\n\n"
        
        plan += f"""## Suggested File Structure

```
{name.replace(' ', '_').lower()}/
{file_structure}
```

## Development Milestones

- [ ] Project setup and initial structure
- [ ] Core functionality implementation
- [ ] User interface/interaction layer
- [ ] Testing and validation
- [ ] Documentation and README
- [ ] Final review and optimization

## Resources and Dependencies

Based on the technology stack, you may need:
{self._suggest_dependencies(tech_stack.lower())}

---
*Generated by Demonology Project Planning Tool*
"""
        
        return plan
    
    def _get_phase_details(self, phase: str, complexity: str) -> str:
        """Get detailed description for each phase."""
        details = {
            "Planning": "- Define project scope and requirements\n- Set up development environment\n- Create project timeline",
            "Research": "- Research existing solutions\n- Evaluate libraries and frameworks\n- Prototype key features",
            "Design": "- Create system architecture\n- Design user interface mockups\n- Plan database schema (if applicable)",
            "Implementation": "- Set up project structure\n- Implement core functionality\n- Build user interface",
            "Testing": "- Write unit tests\n- Perform integration testing\n- Manual testing and bug fixes",
            "Documentation": "- Write API documentation\n- Create user guides\n- Add inline code comments",
            "Deployment": "- Set up production environment\n- Configure CI/CD pipeline\n- Deploy and monitor application"
        }
        return details.get(phase, f"- Complete {phase.lower()} phase")
    
    def _suggest_file_structure(self, tech_stack: str, complexity: str) -> str:
        """Suggest file structure based on technology stack."""
        if "python" in tech_stack:
            return """├── src/
│   ├── __init__.py
│   ├── main.py
│   └── modules/
├── tests/
│   └── test_main.py
├── requirements.txt
├── README.md
└── setup.py"""
        elif "javascript" in tech_stack or "node" in tech_stack:
            return """├── src/
│   ├── index.js
│   └── components/
├── tests/
├── package.json
├── README.md
└── .gitignore"""
        elif "c++" in tech_stack or "cpp" in tech_stack:
            return """├── src/
│   ├── main.cpp
│   └── headers/
├── tests/
├── Makefile
├── README.md
└── CMakeLists.txt"""
        elif "react" in tech_stack:
            return """├── public/
├── src/
│   ├── components/
│   ├── App.js
│   └── index.js
├── package.json
├── README.md
└── .gitignore"""
        else:
            return """├── src/
├── tests/
├── docs/
├── README.md
└── .gitignore"""
    
    def _suggest_dependencies(self, tech_stack: str) -> str:
        """Suggest dependencies based on technology stack."""
        if "python" in tech_stack:
            return "- Python 3.8+\n- pip for package management\n- Virtual environment (venv/conda)\n- Testing framework (pytest)"
        elif "javascript" in tech_stack or "node" in tech_stack:
            return "- Node.js and npm\n- Testing framework (Jest/Mocha)\n- Linting tools (ESLint)"
        elif "c++" in tech_stack:
            return "- C++ compiler (GCC/Clang)\n- Build system (Make/CMake)\n- Testing framework (Google Test)"
        elif "react" in tech_stack:
            return "- Node.js and npm\n- React development tools\n- Testing library (React Testing Library)"
        else:
            return "- Development environment setup\n- Version control (Git)\n- Testing framework\n- Documentation tools"
    
    async def _execute_project_plan(self, project_name: str, tech_stack: str, complexity: str) -> List[str]:
        """Create the actual project structure and files."""
        created_files = []
        
        # Clean up project name for directory use
        clean_name = project_name.replace(' ', '_').lower()
        
        # Create project in current working directory
        try:
            current_dir = Path.cwd().resolve()
            
            # If project_name looks like a path, use it directly
            if "/" in project_name or "\\" in project_name:
                # Try to interpret as relative path from current directory
                project_dir = current_dir / project_name
            else:
                project_dir = current_dir / clean_name
                
        except Exception:
            # If we can't get current directory, fall back to home directory
            if "/" in project_name or "\\" in project_name:
                try:
                    project_dir = Path(project_name).resolve()
                except (PermissionError, ValueError):
                    project_dir = Path.home() / clean_name
            else:
                project_dir = Path.home() / clean_name
        
        # Create project directory
        project_dir.mkdir(parents=True, exist_ok=True)
        created_files.append(str(project_dir))
        
        # Create technology-specific project structure
        if "python" in tech_stack.lower():
            created_files.extend(await self._create_python_project(project_dir, project_name, complexity))
        elif "c++" in tech_stack.lower() or "cpp" in tech_stack.lower():
            created_files.extend(await self._create_cpp_project(project_dir, project_name, complexity))
        elif "javascript" in tech_stack.lower() or "node" in tech_stack.lower():
            created_files.extend(await self._create_js_project(project_dir, project_name, complexity))
        elif "react" in tech_stack.lower():
            created_files.extend(await self._create_react_project(project_dir, project_name, complexity))
        else:
            created_files.extend(await self._create_generic_project(project_dir, project_name, complexity))
        
        return created_files
    
    async def _create_python_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a Python project structure."""
        created = []
        
        # Create directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        tests_dir = project_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        created.append(str(tests_dir))
        
        # Create main Python files
        main_py = src_dir / "main.py"
        main_py.write_text(f'''#!/usr/bin/env python3
"""
{project_name} - Main application entry point.
"""

def main():
    """Main function."""
    print("Hello from {project_name}!")
    # TODO: Implement your application logic here

if __name__ == "__main__":
    main()
''', encoding="utf-8")
        created.append(str(main_py))
        
        # Create __init__.py
        init_py = src_dir / "__init__.py"
        init_py.write_text('"""Main package."""\n', encoding="utf-8")
        created.append(str(init_py))
        
        # Create requirements.txt
        req_txt = project_dir / "requirements.txt"
        req_txt.write_text("# Add your dependencies here\n", encoding="utf-8")
        created.append(str(req_txt))
        
        # Create README.md
        readme = project_dir / "README.md"
        readme.write_text(f'''# {project_name}

## Description
TODO: Add project description

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

## Testing
```bash
python -m pytest tests/
```
''', encoding="utf-8")
        created.append(str(readme))
        
        # Create test file
        test_main = tests_dir / "test_main.py"
        test_main.write_text(f'''"""
Tests for {project_name}.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import main

def test_main():
    """Test main function runs without error."""
    # TODO: Add meaningful tests
    assert True
''', encoding="utf-8")
        created.append(str(test_main))
        
        return created
    
    async def _create_cpp_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a C++ project structure."""
        created = []
        
        # Create directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        # Create main.cpp
        main_cpp = src_dir / "main.cpp"
        main_cpp.write_text(f'''#include <iostream>
#include <string>

int main() {{
    std::cout << "Hello from {project_name}!" << std::endl;
    // TODO: Implement your application logic here
    return 0;
}}
''', encoding="utf-8")
        created.append(str(main_cpp))
        
        # Create Makefile
        makefile = project_dir / "Makefile"
        makefile.write_text(f'''CXX=g++
CXXFLAGS=-std=c++17 -Wall -Wextra -O2
TARGET={project_name.replace(' ', '_').lower()}
SRCDIR=src
SOURCES=$(wildcard $(SRCDIR)/*.cpp)

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean
''', encoding="utf-8")
        created.append(str(makefile))
        
        # Create README.md
        readme = project_dir / "README.md"
        readme.write_text(f'''# {project_name}

## Description
TODO: Add project description

## Build
```bash
make
```

## Run
```bash
./{project_name.replace(' ', '_').lower()}
```

## Clean
```bash
make clean
```
''', encoding="utf-8")
        created.append(str(readme))
        
        return created
    
    async def _create_js_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a JavaScript/Node.js project structure."""
        created = []
        
        # Create directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        # Create main.js
        main_js = src_dir / "index.js"
        main_js.write_text(f'''#!/usr/bin/env node
/**
 * {project_name} - Main application entry point
 */

function main() {{
    console.log("Hello from {project_name}!");
    // TODO: Implement your application logic here
}}

if (require.main === module) {{
    main();
}}

module.exports = {{ main }};
''', encoding="utf-8")
        created.append(str(main_js))
        
        # Create package.json
        package_json = project_dir / "package.json"
        package_json.write_text(f'''{{
  "name": "{project_name.replace(' ', '-').lower()}",
  "version": "1.0.0",
  "description": "TODO: Add project description",
  "main": "src/index.js",
  "scripts": {{
    "start": "node src/index.js",
    "test": "jest"
  }},
  "keywords": [],
  "author": "",
  "license": "MIT",
  "devDependencies": {{
    "jest": "^29.0.0"
  }}
}}
''', encoding="utf-8")
        created.append(str(package_json))
        
        # Create README.md
        readme = project_dir / "README.md"
        readme.write_text(f'''# {project_name}

## Description
TODO: Add project description

## Installation
```bash
npm install
```

## Usage
```bash
npm start
```

## Testing
```bash
npm test
```
''', encoding="utf-8")
        created.append(str(readme))
        
        return created
    
    async def _create_react_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a React project structure."""
        created = []
        
        # Create directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        public_dir = project_dir / "public"
        public_dir.mkdir(exist_ok=True)
        created.append(str(public_dir))
        
        # Create App.js
        app_js = src_dir / "App.js"
        app_js.write_text(f'''import React from 'react';
import './App.css';

function App() {{
  return (
    <div className="App">
      <header className="App-header">
        <h1>{project_name}</h1>
        <p>Welcome to your new React application!</p>
      </header>
    </div>
  );
}}

export default App;
''', encoding="utf-8")
        created.append(str(app_js))
        
        # Create index.js
        index_js = src_dir / "index.js"
        index_js.write_text('''import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
''', encoding="utf-8")
        created.append(str(index_js))
        
        # Create basic CSS
        app_css = src_dir / "App.css"
        app_css.write_text('''.App {
  text-align: center;
}

.App-header {
  background-color: #282c34;
  padding: 20px;
  color: white;
  min-height: 50vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
''', encoding="utf-8")
        created.append(str(app_css))
        
        index_css = src_dir / "index.css"
        index_css.write_text('''body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
}
''', encoding="utf-8")
        created.append(str(index_css))
        
        # Create index.html
        index_html = public_dir / "index.html"
        index_html.write_text(f'''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{project_name}</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
''', encoding="utf-8")
        created.append(str(index_html))
        
        # Create package.json
        package_json = project_dir / "package.json"
        package_json.write_text(f'''{{
  "name": "{project_name.replace(' ', '-').lower()}",
  "version": "0.1.0",
  "private": true,
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  }},
  "scripts": {{
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }},
  "browserslist": {{
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }}
}}
''', encoding="utf-8")
        created.append(str(package_json))
        
        return created
    
    async def _create_generic_project(self, project_dir: Path, project_name: str, complexity: str) -> List[str]:
        """Create a generic project structure."""
        created = []
        
        # Create basic directories
        src_dir = project_dir / "src"
        src_dir.mkdir(exist_ok=True)
        created.append(str(src_dir))
        
        docs_dir = project_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        created.append(str(docs_dir))
        
        # Create README.md
        readme = project_dir / "README.md"
        readme.write_text(f'''# {project_name}

## Description
TODO: Add project description

## Getting Started
TODO: Add setup and usage instructions

## Contributing
TODO: Add contribution guidelines
''', encoding="utf-8")
        created.append(str(readme))
        
        # Create basic source file
        main_file = src_dir / "main.txt"
        main_file.write_text(f"Main file for {project_name}\nTODO: Implement your project here\n", encoding="utf-8")
        created.append(str(main_file))
        
        return created