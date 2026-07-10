import argparse
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Any

def convert_cvefixes(sqlite_db_path: str, output_path: str):
    """
    Converts a CVEfixes SQLite database into the CEVuD benchmark manifest format.
    The benchmark manifest requires grouping by project and specifying git sources.
    """
    conn = sqlite3.connect(sqlite_db_path)
    
    # Typical CVEfixes schema query to get vulnerable and safe functions
    # Adjust query based on the exact version of the CVEfixes schema
    query = """
        SELECT 
            r.repo_url, 
            c.hash AS commit_hash, 
            f.file_path, 
            m.name AS function_name, 
            m.signature, 
            m.nloc,
            m.start_line, 
            m.end_line, 
            m.code,
            cwe.cwe_id
        FROM method_change m
        JOIN file_change f ON m.file_change_id = f.file_change_id
        JOIN commits c ON f.hash = c.hash
        JOIN repository r ON c.repo_url = r.repo_url
        LEFT JOIN cve cve ON c.hash = cve.hash
        LEFT JOIN cwe cwe ON cve.cve_id = cwe.cve_id
        WHERE f.programming_language = 'Python'
    """
    
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Failed to query database, ensure you are using a valid CVEfixes SQLite DB: {e}")
        conn.close()
        return

    conn.close()
    
    # Process into benchmark manifest format
    projects: Dict[str, Dict[str, Any]] = {}
    
    for idx, row in df.iterrows():
        repo_url = row['repo_url']
        project_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        commit_hash = row['commit_hash']
        
        # We will assume a 'label' column exists or derive it (1 for vulnerable, 0 for safe)
        label = 1 if 'vul' in str(row['cwe_id']).lower() or row['cwe_id'] else 0
        
        if project_name not in projects:
            projects[project_name] = {
                "project": project_name,
                "git_source": {
                    "git_url": repo_url,
                    "ref": commit_hash
                },
                "samples": []
            }
        
        sample_id = f"{project_name}::{row['function_name']}::{row['start_line']}"
        
        sample = {
            "sample_id": sample_id,
            "file_path": row['file_path'],
            "function_name": row['function_name'],
            "start_line": int(row['start_line']) if pd.notna(row['start_line']) else 1,
            "end_line": int(row['end_line']) if pd.notna(row['end_line']) else 10,
            "label": label,
            "vulnerability_type": row['cwe_id'] if pd.notna(row['cwe_id']) else "unknown"
        }
        projects[project_name]["samples"].append(sample)
        
    manifest = list(projects.values())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Successfully converted CVEfixes data into {output_path}")
    print(f"Total projects: {len(manifest)}")
    print(f"Total samples: {sum(len(p['samples']) for p in manifest)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CVEfixes SQLite DB to CEVuD benchmark manifest")
    parser.add_argument("--db", required=True, help="Path to CVEfixes SQLite database")
    parser.add_argument("--output", required=True, help="Path to output manifest JSON")
    args = parser.parse_args()
    
    convert_cvefixes(args.db, args.output)
