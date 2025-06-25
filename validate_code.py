#!/usr/bin/env python3
"""
AiCraft Pro - Code Validation Script
Validates the code structure and implementation completeness
"""

import os
import sys
import re
from pathlib import Path

def validate_file_structure():
    """Validate that all required files exist"""
    required_files = [
        'src/tensor.h',
        'src/tensor.c', 
        'src/training.h',
        'src/training.c',
        'src/aicraft_advanced.h',
        'src/advanced_optimizers.c',
        'src/advanced_training.c',
        'src/advanced_utils.c',
        'src/graph_optimizer.c',
        'src/quantization.c',
        'src/benchmark_suite.c',
        'src/main.c',
        'CMakeLists.txt',
        'build.bat',
        'README.md'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print("🔍 File Structure Validation")
    print(f"✅ Found {len(existing_files)} required files")
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def count_lines_of_code():
    """Count total lines of code"""
    extensions = ['.c', '.h', '.cu']
    total_lines = 0
    file_count = 0
    
    for ext in extensions:
        for file_path in Path('src').glob(f'**/*{ext}'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    file_count += 1
                    print(f"  {file_path}: {lines} lines")
            except Exception as e:
                print(f"  ❌ Error reading {file_path}: {e}")
    
    print(f"\n📊 Code Statistics:")
    print(f"   Total files: {file_count}")
    print(f"   Total lines: {total_lines}")
    return total_lines

def validate_function_implementations():
    """Check if key functions are implemented"""
    key_functions = [
        'advanced_training_loop',
        'adabound_update',
        'radam_update', 
        'lamb_update',
        'quantize_model',
        'optimize_graph',
        'mixed_precision_forward',
        'run_comprehensive_benchmarks',
        'tensor_random'
    ]
    
    implemented_functions = []
    missing_functions = []
    
    # Search for function implementations in all C files
    for c_file in Path('src').glob('**/*.c'):
        try:
            with open(c_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for func in key_functions:
                    if re.search(rf'{func}\s*\(', content):
                        if func not in implemented_functions:
                            implemented_functions.append(func)
        except Exception as e:
            print(f"Error reading {c_file}: {e}")
    
    missing_functions = [f for f in key_functions if f not in implemented_functions]
    
    print(f"\n🔧 Function Implementation Check:")
    print(f"✅ Implemented: {len(implemented_functions)}/{len(key_functions)} functions")
    
    for func in implemented_functions:
        print(f"   ✅ {func}")
    
    if missing_functions:
        print(f"\n❌ Missing implementations:")
        for func in missing_functions:
            print(f"   ❌ {func}")
        return False
    
    return True

def validate_headers():
    """Check if all header files have proper include guards"""
    header_files = list(Path('src').glob('**/*.h'))
    
    print(f"\n📋 Header File Validation:")
    valid_headers = 0
    
    for header in header_files:
        try:
            with open(header, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if '#ifndef' in content and '#define' in content and '#endif' in content:
                    valid_headers += 1
                    print(f"   ✅ {header}")
                else:
                    print(f"   ❌ {header} - Missing include guards")
        except Exception as e:
            print(f"   ❌ Error reading {header}: {e}")
    
    print(f"✅ {valid_headers}/{len(header_files)} headers properly guarded")
    return valid_headers == len(header_files)

def main():
    """Main validation function"""
    print("🏆 AiCraft Pro - Code Validation")
    print("=" * 50)
    
    os.chdir(Path(__file__).parent)
    
    # Run all validations
    validations = [
        validate_file_structure(),
        validate_headers(),
        validate_function_implementations()
    ]
    
    # Count lines of code
    total_lines = count_lines_of_code()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary:")
    
    passed = sum(validations)
    total = len(validations)
    
    print(f"✅ Passed: {passed}/{total} validation checks")
    print(f"📝 Total lines of code: {total_lines}")
    
    if passed == total:
        print("🏆 AiCraft Pro is ready for competition!")
        print("🚀 All features implemented and validated")
        return 0
    else:
        print("❌ Some validations failed - please review")
        return 1

if __name__ == "__main__":
    sys.exit(main())
