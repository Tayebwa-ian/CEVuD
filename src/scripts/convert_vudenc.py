import argparse
import json
import pandas as pd
from typing import Dict, List, Any

def convert_vudenc(json_path: str, output_path: str):
    """
    Converts a VUDENC published dataset (JSON format) into the CEVuD benchmark manifest format.
    """
    try:
        # Assuming VUDENC format is a JSON array of vulnerable/safe functions
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load VUDENC JSON: {e}")
        return

    # Process into benchmark manifest format
    projects: Dict[str, Dict[str, Any]] = {}
    
    for item in data:
        # VUDENC has project names, sometimes git repos, file paths, etc.
        # We'll map standard typical fields
        project_name = item.get('project', 'unknown_project')
        repo_url = item.get('repo_url', f"https://github.com/vudenc/{project_name}.git")
        commit_hash = item.get('commit_id', 'HEAD')
        
        # 1 for vulnerable, 0 for safe (VUDENC typically uses 'vul' flag or similar)
        label = 1 if item.get('vul', False) or item.get('target', 0) == 1 else 0
        
        if project_name not in projects:
            projects[project_name] = {
                "project": project_name,
                "git_source": {
                    "git_url": repo_url,
                    "ref": commit_hash
                },
                "samples": []
            }
        
        func_name = item.get('func_name', item.get('function_name', 'unknown_function'))
        start_line = int(item.get('start_line', 1))
        end_line = int(item.get('end_line', 10))
        
        sample_id = f"{project_name}::{func_name}::{start_line}"
        
        sample = {
            "sample_id": sample_id,
            "file_path": item.get('file_path', 'unknown_file.py'),
            "function_name": func_name,
            "start_line": start_line,
            "end_line": end_line,
            "label": label,
            "vulnerability_type": item.get('cwe', 'unknown')
        }
        projects[project_name]["samples"].append(sample)
        
    manifest = list(projects.values())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Successfully converted VUDENC data into {output_path}")
    print(f"Total projects: {len(manifest)}")
    print(f"Total samples: {sum(len(p['samples']) for p in manifest)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VUDENC JSON to CEVuD benchmark manifest")
    parser.add_argument("--input", required=True, help="Path to VUDENC JSON file")
    parser.add_argument("--output", required=True, help="Path to output manifest JSON")
    args = parser.parse_args()
    
    convert_vudenc(args.input, args.output)
