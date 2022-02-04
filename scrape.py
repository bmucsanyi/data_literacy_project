"""Interface to our IMDb scraper."""
from scrapy.crawler import CrawlerProcess


def main():
    process = CrawlerProcess(settings={
        "BOT_NAME": "imdb_spider",
        "SPIDER_MODULES": "src.imdb_scraper.imdb_spider.spiders",
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 1.5,
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
    })
    process.crawl("imdb")
    process.start()

if __name__ == "__main__":
    main()
