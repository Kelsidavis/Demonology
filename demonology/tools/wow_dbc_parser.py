# demonology/tools/wow_dbc_parser.py
from __future__ import annotations

import csv
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import Tool, _confine

class WoWDBCParserTool(Tool):
    """
    World of Warcraft DBC database file parser.
    Parses DBC files and converts them to JSON/CSV for Unreal Engine data tables.
    """

    def __init__(self):
        super().__init__("wow_dbc_parser", "Parse World of Warcraft DBC database files")

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["parse", "batch_parse", "info", "export_datatable", "convert_to_json", "convert_to_csv"],
                        "description": "Operation to perform on DBC files"
                    },
                    "input_path": {
                        "type": "string",
                        "description": "Path to DBC file or directory containing DBC files"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output directory for parsed data",
                        "default": "./parsed_dbc"
                    },
                    "dbc_type": {
                        "type": "string",
                        "enum": ["auto", "item", "spell", "creature", "gameobject", "quest", "achievement", "talent", "skillline"],
                        "description": "Type of DBC file for specialized parsing",
                        "default": "auto"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["json", "csv", "unreal_datatable"],
                        "description": "Output format for parsed data",
                        "default": "json"
                    },
                    "string_encoding": {
                        "type": "string",
                        "enum": ["utf-8", "latin1", "cp1252"],
                        "description": "Character encoding for string data",
                        "default": "utf-8"
                    },
                    "include_strings": {
                        "type": "boolean",
                        "description": "Include string table data in output",
                        "default": True
                    }
                },
                "required": ["operation", "input_path"]
            }
        }

    # DBC field definitions for known file types
    DBC_SCHEMAS = {
        "item": [
            {"name": "ID", "type": "uint32"},
            {"name": "Class", "type": "uint32"},
            {"name": "SubClass", "type": "uint32"},
            {"name": "Sound_Override_Subclass", "type": "int32"},
            {"name": "Material", "type": "uint32"},
            {"name": "DisplayInfoID", "type": "uint32"},
            {"name": "InventoryType", "type": "uint32"},
            {"name": "Sheath", "type": "uint32"}
        ],
        "spell": [
            {"name": "ID", "type": "uint32"},
            {"name": "School", "type": "uint32"},
            {"name": "Category", "type": "uint32"},
            {"name": "CastUI", "type": "uint32"},
            {"name": "Dispel", "type": "uint32"},
            {"name": "Mechanic", "type": "uint32"},
            {"name": "Attributes", "type": "uint32"},
            {"name": "AttributesEx", "type": "uint32"}
        ],
        "creature": [
            {"name": "ID", "type": "uint32"},
            {"name": "Name", "type": "string"},
            {"name": "SubName", "type": "string"},
            {"name": "IconName", "type": "string"},
            {"name": "TypeFlags", "type": "uint32"},
            {"name": "Type", "type": "uint32"},
            {"name": "Family", "type": "uint32"},
            {"name": "Rank", "type": "uint32"}
        ]
    }

    def _parse_dbc_header(self, data: bytes) -> Dict[str, Any]:
        """Parse DBC file header."""
        if len(data) < 20:
            raise ValueError("Invalid DBC file: too short")
        
        # DBC header: signature(4), record_count(4), field_count(4), record_size(4), string_block_size(4)
        header = struct.unpack('<4sIIII', data[:20])
        
        signature = header[0]
        if signature != b'WDBC':
            raise ValueError(f"Invalid DBC signature: {signature}")
        
        return {
            "signature": signature.decode('ascii'),
            "record_count": header[1],
            "field_count": header[2],
            "record_size": header[3],
            "string_block_size": header[4],
            "header_size": 20
        }

    def _parse_string_block(self, data: bytes, offset: int, size: int, encoding: str = "utf-8") -> Dict[int, str]:
        """Parse the string block from DBC file."""
        string_block = data[offset:offset + size]
        strings = {}
        
        current_offset = 0
        current_string = b""
        
        for i, byte in enumerate(string_block):
            if byte == 0:  # Null terminator
                if current_string:
                    try:
                        strings[current_offset] = current_string.decode(encoding, errors='replace')
                    except UnicodeDecodeError:
                        strings[current_offset] = current_string.decode('latin1', errors='replace')
                current_offset = i + 1
                current_string = b""
            else:
                if not current_string:  # Start of new string
                    current_offset = i
                current_string += bytes([byte])
        
        return strings

    def _parse_records(self, data: bytes, header: Dict[str, Any], strings: Dict[int, str],
                      schema: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """Parse DBC records."""
        records = []
        record_start = header["header_size"]
        record_size = header["record_size"]
        record_count = header["record_count"]
        field_count = header["field_count"]
        
        for i in range(record_count):
            record_offset = record_start + (i * record_size)
            record_data = data[record_offset:record_offset + record_size]
            
            if len(record_data) < record_size:
                break
            
            record = {}
            
            # Parse fields based on schema or as generic uint32 values
            if schema and len(schema) <= field_count:
                field_offset = 0
                for j, field_def in enumerate(schema):
                    field_name = field_def["name"]
                    field_type = field_def["type"]
                    
                    if field_offset + 4 > len(record_data):
                        break
                    
                    if field_type == "uint32":
                        value = struct.unpack('<I', record_data[field_offset:field_offset + 4])[0]
                    elif field_type == "int32":
                        value = struct.unpack('<i', record_data[field_offset:field_offset + 4])[0]
                    elif field_type == "float":
                        value = struct.unpack('<f', record_data[field_offset:field_offset + 4])[0]
                    elif field_type == "string":
                        string_offset = struct.unpack('<I', record_data[field_offset:field_offset + 4])[0]
                        value = strings.get(string_offset, f"<string_{string_offset}>")
                    else:
                        value = struct.unpack('<I', record_data[field_offset:field_offset + 4])[0]
                    
                    record[field_name] = value
                    field_offset += 4
            else:
                # Generic parsing - treat all fields as uint32
                for j in range(min(field_count, len(record_data) // 4)):
                    field_offset = j * 4
                    value = struct.unpack('<I', record_data[field_offset:field_offset + 4])[0]
                    
                    # Try to resolve as string if it looks like a string offset
                    if value in strings and value < header["string_block_size"]:
                        record[f"field_{j}"] = strings[value]
                    else:
                        record[f"field_{j}"] = value
            
            records.append(record)
        
        return records

    def _detect_dbc_type(self, filename: str) -> Optional[str]:
        """Detect DBC type from filename."""
        filename_lower = filename.lower()
        
        if "item" in filename_lower:
            return "item"
        elif "spell" in filename_lower:
            return "spell"
        elif "creature" in filename_lower:
            return "creature"
        elif "gameobject" in filename_lower:
            return "gameobject"
        elif "quest" in filename_lower:
            return "quest"
        elif "achievement" in filename_lower:
            return "achievement"
        elif "talent" in filename_lower:
            return "talent"
        elif "skill" in filename_lower:
            return "skillline"
        
        return None

    def _generate_unreal_datatable(self, records: List[Dict[str, Any]], table_name: str) -> str:
        """Generate Unreal Engine DataTable format."""
        if not records:
            return "{}"
        
        # Analyze field types from first record
        sample_record = records[0]
        field_types = {}
        
        for field_name, value in sample_record.items():
            if isinstance(value, int):
                if value >= 0 and value <= 4294967295:  # uint32 range
                    field_types[field_name] = "uint32"
                else:
                    field_types[field_name] = "int32"
            elif isinstance(value, float):
                field_types[field_name] = "float"
            elif isinstance(value, str):
                field_types[field_name] = "FString"
            else:
                field_types[field_name] = "FString"
        
        # Generate DataTable structure
        datatable = {
            "Type": "DataTable",
            "Name": table_name,
            "RowStruct": f"F{table_name}Row",
            "StructDefinition": {
                "Fields": [
                    {
                        "Name": field_name,
                        "Type": field_types[field_name],
                        "Description": f"Field {field_name} from DBC"
                    }
                    for field_name in sample_record.keys()
                ]
            },
            "Rows": {}
        }
        
        # Add all records as rows
        for i, record in enumerate(records):
            row_name = f"Row_{record.get('ID', i)}"
            datatable["Rows"][row_name] = record
        
        return json.dumps(datatable, indent=2)

    async def _parse_dbc_file(self, input_file: Path, dbc_type: str, output_format: str,
                            string_encoding: str, include_strings: bool) -> Dict[str, Any]:
        """Parse a single DBC file."""
        try:
            with open(input_file, 'rb') as f:
                dbc_data = f.read()
            
            # Parse header
            header = self._parse_dbc_header(dbc_data)
            
            # Parse string block
            string_block_offset = (header["header_size"] + 
                                 header["record_count"] * header["record_size"])
            strings = {}
            
            if include_strings and header["string_block_size"] > 0:
                strings = self._parse_string_block(
                    dbc_data, string_block_offset, 
                    header["string_block_size"], string_encoding
                )
            
            # Determine schema
            if dbc_type == "auto":
                dbc_type = self._detect_dbc_type(input_file.name) or "generic"
            
            schema = self.DBC_SCHEMAS.get(dbc_type)
            
            # Parse records
            records = self._parse_records(dbc_data, header, strings, schema)
            
            return {
                "success": True,
                "input_file": str(input_file),
                "dbc_type": dbc_type,
                "header": header,
                "record_count": len(records),
                "string_count": len(strings),
                "records": records,
                "strings": strings if include_strings else {}
            }
            
        except Exception as e:
            return {
                "success": False,
                "input_file": str(input_file),
                "error": str(e)
            }

    async def execute(self, operation: str, input_path: str, output_path: str = "./parsed_dbc",
                     dbc_type: str = "auto", output_format: str = "json",
                     string_encoding: str = "utf-8", include_strings: bool = True, **_) -> Dict[str, Any]:
        
        try:
            input_path_obj = _confine(Path(input_path))
            output_dir = _confine(Path(output_path))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if operation == "parse" or operation == "convert_to_json" or operation == "convert_to_csv":
                if not input_path_obj.exists():
                    return {"success": False, "error": f"Input file not found: {input_path}"}
                
                if not input_path_obj.suffix.lower() == '.dbc':
                    return {"success": False, "error": "Input file must be a DBC file"}
                
                result = await self._parse_dbc_file(
                    input_path_obj, dbc_type, output_format, string_encoding, include_strings
                )
                
                if result["success"]:
                    file_stem = input_path_obj.stem
                    
                    if operation == "convert_to_json" or output_format == "json":
                        output_file = output_dir / f"{file_stem}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                "header": result["header"],
                                "records": result["records"],
                                "strings": result["strings"]
                            }, f, indent=2, ensure_ascii=False)
                        result["output_file"] = str(output_file)
                    
                    elif operation == "convert_to_csv" or output_format == "csv":
                        output_file = output_dir / f"{file_stem}.csv"
                        if result["records"]:
                            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                                writer = csv.DictWriter(f, fieldnames=result["records"][0].keys())
                                writer.writeheader()
                                writer.writerows(result["records"])
                        result["output_file"] = str(output_file)
                
                return result
            
            elif operation == "batch_parse":
                if not input_path_obj.exists():
                    return {"success": False, "error": f"Input directory not found: {input_path}"}
                
                dbc_files = list(input_path_obj.rglob("*.dbc"))
                if not dbc_files:
                    return {"success": False, "error": "No DBC files found in directory"}
                
                results = []
                successful = 0
                failed = 0
                
                for dbc_file in dbc_files:
                    result = await self._parse_dbc_file(
                        dbc_file, dbc_type, output_format, string_encoding, include_strings
                    )
                    
                    if result["success"]:
                        # Save parsed data
                        file_stem = dbc_file.stem
                        
                        if output_format == "json":
                            output_file = output_dir / f"{file_stem}.json"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump({
                                    "header": result["header"],
                                    "records": result["records"],
                                    "strings": result["strings"]
                                }, f, indent=2, ensure_ascii=False)
                            result["output_file"] = str(output_file)
                        
                        elif output_format == "csv":
                            output_file = output_dir / f"{file_stem}.csv"
                            if result["records"]:
                                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                                    writer = csv.DictWriter(f, fieldnames=result["records"][0].keys())
                                    writer.writeheader()
                                    writer.writerows(result["records"])
                                result["output_file"] = str(output_file)
                        
                        successful += 1
                    else:
                        failed += 1
                    
                    results.append(result)
                
                return {
                    "success": True,
                    "operation": "batch_parse",
                    "total_files": len(dbc_files),
                    "successful": successful,
                    "failed": failed,
                    "output_directory": str(output_dir),
                    "results": results
                }
            
            elif operation == "info":
                if not input_path_obj.exists() or not input_path_obj.suffix.lower() == '.dbc':
                    return {"success": False, "error": "Invalid DBC file"}
                
                with open(input_path_obj, 'rb') as f:
                    dbc_data = f.read()
                
                header = self._parse_dbc_header(dbc_data)
                detected_type = self._detect_dbc_type(input_path_obj.name)
                
                return {
                    "success": True,
                    "file": str(input_path_obj),
                    "signature": header["signature"],
                    "record_count": header["record_count"],
                    "field_count": header["field_count"],
                    "record_size": header["record_size"],
                    "string_block_size": header["string_block_size"],
                    "detected_type": detected_type,
                    "file_size": len(dbc_data)
                }
            
            elif operation == "export_datatable":
                result = await self._parse_dbc_file(
                    input_path_obj, dbc_type, "json", string_encoding, include_strings
                )
                
                if result["success"]:
                    table_name = input_path_obj.stem.replace('.', '_').title()
                    datatable_content = self._generate_unreal_datatable(result["records"], table_name)
                    
                    datatable_file = output_dir / f"{table_name}_DataTable.uasset"
                    with open(datatable_file, 'w', encoding='utf-8') as f:
                        f.write(datatable_content)
                    
                    result["datatable_file"] = str(datatable_file)
                    result["table_name"] = table_name
                
                return result
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": f"DBC parsing failed: {str(e)}"}