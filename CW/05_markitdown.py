from markitdown import MarkItDown
import os

input_pdf = "example.pdf"
output_md = "output_markitdown.md"

def run():
    if not os.path.exists(input_pdf):
        print(f"錯誤：找不到 {input_pdf}")
        return

    print("正在使用 MarkitDown 轉換文件...")
    md = MarkItDown()
    result = md.convert(input_pdf)
    
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    print(f"完成！結果已儲存至 {output_md}")

if __name__ == "__main__":
    run()