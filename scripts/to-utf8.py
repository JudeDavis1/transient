"""Converts all files in dataset to UTF-8 if not already."""


import chardet
import glob

# Function to convert a file to UTF-8
def convert_to_utf8(filename):
    # Detect the file encoding
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        original_encoding = result['encoding']
    
    # If the detected encoding is not UTF-8, convert the file to UTF-8
    if original_encoding.lower() != 'utf-8':
        try:
            with open(filename, 'r', encoding=original_encoding) as file:
                content = file.read()
            # Write the content to a new file or overwrite the old one
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"Converted {filename} from {original_encoding} to UTF-8.")
        except UnicodeDecodeError:
            print(f"Could not convert {filename} from {original_encoding}.")
        except TypeError:
            print(f"Encoding type error for {filename}.")
    else:
        print(f"{filename} is already UTF-8.")

# Convert all files in the 'data' directory to UTF-8
for file_path in glob.glob("data/*"):
    convert_to_utf8(file_path)
