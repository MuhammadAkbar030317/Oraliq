import os
import logging
from Source.data_loader import scrape_top5_leagues

log_folder = "C:/Users/User/Desktop/AI_Projects/Project_05/Log"
os.makedirs(log_folder, exist_ok=True)

log_file = os.path.join(log_folder, "web_scraping.log")


logging.basicConfig(
    filename=log_file,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("=== Web scraping skripti ishga tushdi ===")

base_path = "C:/Users/User/Desktop/AI_Projects/Project_05/Data"
raw_path = os.path.join(base_path, "Web_scraped")
merged_path = os.path.join(base_path, "Raw_data")

scrape_top5_leagues(raw_path, merged_path)

logging.info("=== Web scraping skripti tugadi ===")
print(f"Log fayli: {log_file}")
