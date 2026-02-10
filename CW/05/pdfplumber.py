import pdfplumber
import os

input_pdf = "example.pdf"
output_md = "output_pdfplumber.md"

def run():
    if not os.path.exists(input_pdf):
        print(f"錯誤：找不到 {input_pdf}")
        return

    print("正在使用 pdfplumber 提取文字...")
    with pdfplumber.open(input_pdf) as pdf:
        content = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                content.append(f"## Page {i+1}\n\n{text}\n\n---\n")
        
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("".join(content))
    print(f"完成！結果已儲存至 {output_md}")

if __name__ == "__main__":
    run(