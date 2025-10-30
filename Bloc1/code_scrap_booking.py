import os
import sys
import json
import subprocess
from collections import defaultdict
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy_playwright.page import PageMethod  

class BookingSpider(scrapy.Spider):
    name = "booking"
    
    custom_settings = {
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'PLAYWRIGHT_LAUNCH_OPTIONS': {"headless": True},
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 30000,  # 30 secondes
        'DOWNLOAD_DELAY': 2.0,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'LOG_LEVEL': 'INFO',
        'RETRY_TIMES': 3,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 10,
    }

    def __init__(self, villes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.villes = villes or []
        self.items_by_city = defaultdict(list)

    def start_requests(self):
        for city in self.villes:
            url = f'https://www.booking.com/searchresults.fr.html?ss={city}%2C+France'
            self.logger.info(f"🌍 Recherche lancée pour {city} → {url}")
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta={
                    'city': city,
                    'playwright': True,
                    'playwright_page_methods': [
                        PageMethod('wait_for_selector', 'div[data-testid="property-card"]'),
                        PageMethod('wait_for_timeout', 5000)  # attendre 5 secondes pour laisser charger
                    ]
                }
            )

    def parse(self, response):
        city = response.meta['city']
        hotels = response.xpath('//div[contains(@data-testid, "property-card")]')
        self.logger.info(f"🏨 {city}: {len(hotels)} hôtels trouvés")

        for list_hotel in hotels:
            name = list_hotel.xpath('.//div[@data-testid="title"]/text()').get()
            name = name.strip() if name else None
            url = list_hotel.xpath('.//a[@data-testid="title-link"]/@href').get()
            rating = list_hotel.xpath('.//div[@data-testid="review-score"]/div/text()').get()

            if url:
                url = response.urljoin(url)
                self.logger.info(f"➡️ Hôtel détecté: {name} ({city}) | Note: {rating}")
                yield response.follow(
                    url, self.parse_hotel,
                    meta={'name': name, 'rating': rating, 'city': city, 'playwright': True}
                )

    def parse_hotel(self, response):
        name = response.meta['name']
        rating = response.meta['rating']
        city = response.meta['city']

        gps = response.xpath('//a[@id="map_trigger_header"]/@data-atlas-latlng').get()
        latitude, longitude = gps.split(',') if gps else (None, None)
        description = response.xpath('//p[@data-testid="property-description"]/text()').get()

        self.logger.info(f"✅ Hôtel sauvegardé: {name} ({city})")

        self.items_by_city[city].append({
            'ville': city,
            'name': name,
            'description': description,
            'url': response.url,
            'rating': rating,
            'latitude': latitude,
            'longitude': longitude
        })

    def closed(self, reason):
        """Sauvegarde des résultats en JSON par ville"""
        output_dir = "booking_results"
        os.makedirs(output_dir, exist_ok=True)

        for city, items in self.items_by_city.items():
            filename = os.path.join(output_dir, f"results_{city.lower().replace(' ', '_')}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=4)
            self.logger.info(f"💾 Résultats sauvegardés dans {filename}")


if __name__ == "__main__":
    from time import sleep
    VILLES = [
        "Mont Saint Michel", "Saint-Malo", "Bayeux", "Le Havre", "Rouen", "Paris", "Amiens", "Lille",
        "Strasbourg", "Chateau du Haut Koenigsbourg", "Colmar", "Eguisheim", "Besancon", "Dijon",
        "Annecy", "Grenoble", "Lyon", "Gorges du Verdon", "Bormes les Mimosas","Cassis",
        "Marseille","Aix en Provence", "Avignon", "Uzes", "Nimes", "Aigues Mortes",
        "Saintes Maries de la mer", "Collioure", "Carcassonne", "Foix", "Toulouse",
        "Montauban", "Biarritz", "Bayonne", "La Rochelle"
    ]
    BATCH_SIZE = 2
    PAUSE_BATCH = 30  # secondes entre chaque batch

    if len(sys.argv) > 1 and sys.argv[1] != "--main":
        batch = sys.argv[1:]
        process = CrawlerProcess()
        process.crawl(BookingSpider, villes=batch)
        process.start()
    else:
        for i in range(0, len(VILLES), BATCH_SIZE):
            batch = VILLES[i:i+BATCH_SIZE]
            print(f"\n🚀 Lancement batch {i//BATCH_SIZE+1}: {batch}\n")
            subprocess.run([sys.executable, __file__] + batch)
            if i + BATCH_SIZE < len(VILLES):
                print(f"⏳ Pause {PAUSE_BATCH}s avant le prochain batch...")
                sleep(PAUSE_BATCH)