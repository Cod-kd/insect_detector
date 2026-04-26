import re

filepath = 'tflm_standalone/tensorflow/lite/micro/compatibility.h'

with open(filepath, 'r') as f:
    content = f.read()

old = '#define TF_LITE_REMOVE_VIRTUAL_DELETE \\\n  void operator delete(void* p) {}'
new = '#define TF_LITE_REMOVE_VIRTUAL_DELETE \\\n  public: \\\n  void operator delete(void* p) {} \\\n  private:'

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('Patched successfully')
else:
    print('Pattern not found, printing macro area for debug:')
    for i, line in enumerate(content.splitlines()):
        if 'REMOVE_VIRTUAL' in line or 'operator delete' in line:
            print(f'  line {i}: {repr(line)}')