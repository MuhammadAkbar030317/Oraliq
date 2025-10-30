import os
import time
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup


def scrape_top5_leagues(raw_path, merged_path):
    """
    5 ta top liganing (Premier League, La Liga, Bundesliga, Serie A, Ligue 1)
    oxirgi 20 yillik ma'lumotlarini Transfermarkt'dan yuklab olish.
    Har bir liga alohida CSV sifatida saqlanadi, so‘ng birlashtiriladi.
    """

    # 📘 Ligalarning URL manzillari
    leagues = {
        "Premier_League": "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1",
        "La_Liga": "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1",
        "Bundesliga": "https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1",
        "Serie_A": "https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1",
        "Ligue_1": "https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1"
    }

    headers = {"User-Agent": "Mozilla/5.0"}
    years = list(range(2005, 2026))

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(merged_path, exist_ok=True)

    all_leagues = []

    logging.info("Top 5 ligalar uchun scraping jarayoni boshlandi.")

    # 🔁 Har bir liga uchun scraping
    for league_name, base_url in leagues.items():
        logging.info(f"{league_name} uchun ma'lumot yuklanmoqda...")
        league_dfs = []

        for year in years:
            url = f"{base_url}/plus/?saison_id={year}"
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    logging.warning(f"{league_name} - {year}: Sahifa topilmadi! (status: {response.status_code})")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                table = soup.find("table", {"class": "items"})
                if not table:
                    logging.warning(f"{league_name} - {year}: Jadval topilmadi!")
                    continue

                rows = []
                for row in table.find("tbody").find_all("tr"):
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if cols:
                        rows.append(cols)

                if not rows:
                    logging.warning(f"{league_name} - {year}: Ma'lumot topilmadi.")
                    continue

                columns = ["Rank", "Club", "Squad size", "Average age", "Foreigners", "Market value", "Total market value"]
                df = pd.DataFrame(rows, columns=columns)
                df["League"] = league_name
                df["Season"] = year
                league_dfs.append(df)

                logging.info(f"{league_name} - {year}: {len(df)} ta klub ma'lumoti olindi.")
                time.sleep(1)  # serverga yuk tushmasin

            except Exception as e:
                logging.error(f"{league_name} - {year}: Xatolik yuz berdi → {e}")

        # 20 yilni birlashtiramiz
        if league_dfs:
            league_df = pd.concat(league_dfs, ignore_index=True)
            save_file = os.path.join(raw_path, f"{league_name}_20years.csv")
            league_df.to_csv(save_file, index=False)
            logging.info(f"{league_name} ma'lumotlari saqlandi: {save_file}")
            all_leagues.append(league_df)
        else:
            logging.warning(f"{league_name} uchun ma'lumot topilmadi!")

    # 🌍 Barcha ligalarni birlashtirish
    if all_leagues:
        final_df = pd.concat(all_leagues, ignore_index=True)
        final_path = os.path.join(merged_path, "Top5_Leagues_2005_2025.csv")
        final_df.to_csv(final_path, index=False)
        logging.info(f"Barcha ligalar muvaffaqiyatli birlashtirildi → {final_path}")
        print(f"\n✅ Barcha ligalar saqlandi: {final_path}")
    else:
        logging.error("Hech qanday liga uchun ma'lumot topilmadi!")
        print("\n❌ Hech qanday liga uchun ma'lumot topilmadi!")

    logging.info("Scraping jarayoni tugatildi.")
