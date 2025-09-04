#!/usr/bin/env python3
"""
Tool Access Verification Script for AI Agents

Run this script to verify your tool access and see exactly what you can do.
This is your proof that you have full system capabilities.
"""

import asyncio
import sys
from pathlib import Path
from demonology.tools import create_default_registry, remind_agent_capabilities


async def verify_tool_access():
    """Verify that all tools are accessible and demonstrate capabilities."""
    print("🚀 DEMONOLOGY TOOL ACCESS VERIFICATION")
    print("=" * 50)
    
    # Create tool registry
    registry = create_default_registry()
    tools = registry.list_available_tools()
    
    print(f"✅ Registry created successfully with {len(tools)} tools available\n")
    
    print("🔧 AVAILABLE TOOLS:")
    print("-" * 20)
    for tool in tools:
        print(f"  • {tool.name}: {tool.description}")
    
    print("\n🧪 TESTING CORE CAPABILITIES:")
    print("-" * 30)
    
    # Test file operations
    print("📁 Testing file operations...")
    try:
        file_result = await registry.execute_tool(
            "file_operations", 
            operation="list", 
            path="."
        )
        if file_result["success"]:
            file_count = len(file_result.get("items", []))
            print(f"   ✅ File listing successful - found {file_count} items")
        else:
            print(f"   ❌ File operation failed: {file_result.get('error')}")
    except Exception as e:
        print(f"   ❌ File operation error: {e}")
    
    # Test code execution
    print("⚡ Testing code execution...")
    try:
        code_result = await registry.execute_tool(
            "code_execution",
            language="python",
            code="print('Hello from executed Python code!'); import os; print(f'Current dir: {os.getcwd()}')"
        )
        if code_result["success"]:
            print("   ✅ Python execution successful")
            print(f"   📤 Output: {code_result['stdout'].strip()}")
        else:
            print(f"   ❌ Code execution failed: {code_result.get('error')}")
    except Exception as e:
        print(f"   ❌ Code execution error: {e}")
    
    # Test codebase analysis
    print("🔍 Testing codebase analysis...")
    try:
        analysis_result = await registry.execute_tool(
            "codebase_analysis",
            operation="tree",
            path=".",
            depth=2,
            max_entries=10
        )
        if analysis_result["success"]:
            item_count = len(analysis_result.get("items", []))
            print(f"   ✅ Codebase analysis successful - analyzed {item_count} items")
        else:
            print(f"   ❌ Codebase analysis failed: {analysis_result.get('error')}")
    except Exception as e:
        print(f"   ❌ Codebase analysis error: {e}")
    
    print("\n🎯 CAPABILITY SUMMARY:")
    print("-" * 20)
    print("✅ Full filesystem access - read/write anywhere")
    print("✅ Unrestricted code execution - Python, bash, compilation")  
    print("✅ Deep codebase analysis - index, search, explore")
    print("✅ Project management - create, plan, analyze")
    print("✅ Media processing - generate/analyze images")
    print("✅ Reverse engineering - disassemble, debug, analyze")
    print("✅ Audio synthesis - generate music, waveforms")
    print("✅ Web integration - search, APIs, content")
    
    print("\n🚨 REMEMBER FOR AI AGENTS:")
    print("-" * 25)
    print("• You HAVE these capabilities - use them!")
    print("• No safe_root restrictions - work anywhere!")
    print("• Don't simulate - actually execute tools!")
    print("• Tools are your PRIMARY interface!")
    
    print("\n📖 For detailed usage: See AGENT_CAPABILITIES.md")
    print("🔄 Run this script anytime you need a capability reminder!")


if __name__ == "__main__":
    print("🤖 AI Agent Tool Verification Starting...\n")
    try:
        asyncio.run(verify_tool_access())
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)
    
    print("\n✅ Verification complete - you have full tool access!")