import json
import re

def process_json_file(input_file, output_file, pattern_to_remove, replacement):
    """
    读取JSON文件，删除并替换指定字符组合，然后保存结果。

    :param input_file: 输入JSON文件的路径
    :param output_file: 输出JSON文件的路径
    :param pattern_to_remove: 要删除并替换的正则表达式模式
    :param replacement: 替换的字符串
    """
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 将JSON数据转换为字符串
        json_str = json.dumps(data)

        # 使用正则表达式替换指定的字符组合
        processed_str = re.sub(pattern_to_remove, replacement, json_str)

        # 将处理后的字符串转换回JSON对象
        processed_data = json.loads(processed_str)

        # 保存处理后的JSON数据到新的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

        print(f"处理完成，结果已保存到 {output_file}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 示例用法
if __name__ == "__main__":
    input_file_path = 'D:\文件\其他\TEST\\.json备份\\add_0\\备份\\drop_0.json'  # 输入文件路径
    output_file_path = 'D:\文件\其他\TEST\\.json备份\\add_0\\修改\\drop_0.json'  # 输出文件路径
    pattern = r'chenghaonan/qqt'  # 要删除并替换的复杂组合的正则表达式
    replacement_string = 'qiqiutang'  # 想要的替换字符串

    process_json_file(input_file_path, output_file_path, pattern, replacement_string)
