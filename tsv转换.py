import csv

# 定义输入和输出文件的路径
csv_file_path = 'output.csv'
tsv_file_path = 'output.tsv'

# 打开CSV文件进行读取
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)

    # 打开TSV文件进行写入
    with open(tsv_file_path, mode='w', newline='', encoding='utf-8') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')

        # 逐行读取CSV并写入TSV
        for row in csv_reader:
            tsv_writer.writerow(row)

print("CSV文件已成功转换为TSV文件。")