import logging
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.pipeline.vlm_pipeline import VlmPipeline

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_vlm_options() -> ApiVlmOptions:
    """
    配置 VLM 選項。
    針對 ws-02 伺服器與 gemma-3-27b-it 進行優化，
    調低 scale 以避免 Cloudflare 524 超時錯誤。
    """
    api_url = "https://ws-02.wade0426.me/v1/chat/completions"
    
    return ApiVlmOptions(
        url=api_url,
        params={
            "model": "gemma-3-27b-it",
            "max_tokens": 4096,  # 減少生成長度以加快速度
            "temperature": 0.0,
        },
        prompt=(
            "Convert this page to Markdown. Focus on table accuracy. "
            "Output only Markdown code without any explanation."
        ),
        timeout=300, 
        scale=1.0,  # 重要：降至 1.0 以縮短伺服器處理圖像的時間，防止超時
        response_format=ResponseFormat.MARKDOWN,
    )

def run_idp_process():
    # 路徑設定
    cw_dir = Path("/home/pc-49/Desktop/nutc25041lab_hw/CW/")
    input_pdf = cw_dir / "sample_table.pdf"
    output_md = cw_dir / "homework6_output.md"

    if not input_pdf.exists():
        logger.error(f"❌ 找不到輸入檔案：{input_pdf}")
        return

    print("--- [IDP Step 1: 初始化 ws-02 伺服器配置] ---")
    
    # 配置 Pipeline 選項
    vlm_options = get_vlm_options()
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True 
    )

    # 建立轉換器，指定 PDF 使用 VlmPipeline
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline
            )
        }
    )

    print(f"--- [IDP Step 2: 正在處理 {input_pdf.name} (Gemma-3-27B) ...] ---")
    
    try:
        # 開始轉換
        result = doc_converter.convert(input_pdf)
        md_content = result.document.export_to_markdown()

        # 儲存結果
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        print("--- [IDP Step 3: 處理完成] ---")
        print(f"✅ 結果已儲存至: {output_md}")
        
    except Exception as e:
        logger.error(f"❌ 發生錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    run_idp_process()