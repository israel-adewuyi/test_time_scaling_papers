import os
import csv
import cloudscraper

from bs4 import BeautifulSoup
from typing import Optional, List


class GameOf24Scrapper:
    def __init__(self, output_dir: str = "data", sample: bool = True):
        self.base_url = "https://www.4nums.com/game/difficulties/"
        self.sample = sample
        self.output_dir = output_dir
        self.scraper = cloudscraper.create_scraper()  # Use cloudscraper to bypass Cloudflare
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Referer": "https://codeforces.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_page_content(self) -> Optional[BeautifulSoup]:
        try:
            response = self.scraper.get(self.base_url, headers=self.headers)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print(f"Error fetching {self.base_url}:{e}")
            return None

    def scrape(self):
        soup = self.get_page_content()
        if not soup:
            print(f"Failed to get page content from {self.base_url}")
            return

        rows = soup.find_all("tr")
        extracted_info = []

        for row in rows:
            cols = row.find_all("td")
            rank = cols[0].get_text(strip=True)
            puzzle = cols[1].get_text(strip=True)
            extracted_info.append((rank, puzzle))

        # remove the header
        extracted_info = extracted_info[1:]

        # extract sample
        if self.sample:
            extracted_info = [row for row in extracted_info if int(row[0]) > 900 and int(row[0]) <= 1000]
            
        self.save_to_csv(extracted_info)
        # print(extracted_info)

    def save_to_csv(self, data: List[tuple]):
        output_path = os.path.join(self.output_dir, "game_of_24.csv")
        with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Rank", "Puzzle"])
            writer.writerows(data) 
        print(f"Saved {len(data)} records to {output_path}")
        

def main():
    scraper = GameOf24Scrapper()
    scraper.scrape()


if __name__ == "__main__":
    main()
