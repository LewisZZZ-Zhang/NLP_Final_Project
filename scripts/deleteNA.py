import json

# 输入文件路径
input_file = "data/arnor_test.txt"
# 输出文件路径
output_file = "data/arnor_test_without_NA.txt"

def filter_na_relations(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                # 解析 JSON 数据
                data = json.loads(line)
                # 检查 relation 是否为 NA
                if data.get("relation") != "NA":
                    # 写入到输出文件
                    outfile.write(json.dumps(data) + "\n")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")

if __name__ == "__main__":
    filter_na_relations(input_file, output_file)
    print(f"Filtered data saved to {output_file}")

# 使用例
# python3 data_process/1.py