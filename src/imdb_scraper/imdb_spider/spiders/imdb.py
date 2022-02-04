import scrapy
import pandas as pd


class ImdbSpider(scrapy.Spider):
    name = "imdb"
    allowed_domains = ["imdb.com"]

    def __init__(self):
        url_body = "https://www.imdb.com/title/"
        tconst_data = pd.read_csv("../dat/tconst.csv", squeeze=True)
        self.start_urls = [
            f"{url_body}{entry}/" for entry in tconst_data.tolist()
        ]  # [::-1000][:10]
        self.buffer = []

    def parse(self, response):
        tconst = response.url.split("/")[-2]

        resp_dict = {
            "tconst": tconst,
            "Budget": None,
            "Gross US & Canada": None,
            "Opening weekend US & Canada": None,
            "Gross worldwide": None,
        }

        box_office = response.xpath(
            '//ul[@class="ipc-metadata-list ipc-metadata-list--dividers-none '
            "ipc-metadata-list--compact BoxOffice__MetaDataListBoxOffice-sc-40s2pl-0 "
            'fxZYqi ipc-metadata-list--base"]'
        )

        if box_office is not None:
            element = box_office.xpath(
                './/li[@class="ipc-metadata-list__item BoxOffice__'
                'MetaDataListItemBoxOffice-sc-40s2pl-2 gwNUHl"]'
            )

            for elem in element:
                name = elem.xpath(
                    './/span[@class="ipc-metadata-list-item__label"]/text()'
                ).get()
                value = elem.xpath(
                    './/span[@class="ipc-metadata-list-item__list-content-item"]/text()'
                ).get()

                resp_dict[name] = value

        rating = response.xpath(
            '//a[@class="ipc-link ipc-link--baseAlt ipc-link--inherit-color '
            'TitleBlockMetaData__StyledTextLink-sc-12ein40-1 rgaOW"]/text()'
        ).getall()
        if len(rating) > 1:
            rating = rating[1]
        else:
            rating = None
        resp_dict["Rating"] = rating

        critic_elements = response.xpath(
            '//a[@class="ipc-link ipc-link--baseAlt ipc-link--touch-target '
            'ReviewContent__StyledTextLink-sc-vlmc3o-2 dTjvFT isReview"]'
        )
        if critic_elements:
            user_label = (
                critic_elements[0].xpath('.//span[@class="label"]/text()').get()
            )
            user_reviews = (
                critic_elements[0].xpath('.//span[@class="score"]/text()').get()
            )
            resp_dict["User reviews"] = (
                user_reviews if user_label == "User reviews" else None
            )

            if len(critic_elements) > 1:
                critic_label = (
                    critic_elements[1].xpath('.//span[@class="label"]/text()').get()
                )
                critic_reviews = (
                    critic_elements[1].xpath('.//span[@class="score"]/text()').get()
                )
                resp_dict["Critic reviews"] = (
                    critic_reviews if critic_label == "Critic reviews" else None
                )
            else:
                resp_dict["Critic reviews"] = None
        else:
            resp_dict["Critic reviews"] = None
            resp_dict["User reviews"] = None

        self.buffer.append(resp_dict)
        self.check_save(response)

    def check_save(self, response):
        if len(self.buffer) > 2_000 or self.start_urls[-1] == response.url:
            new_df = pd.DataFrame(self.buffer)
            try:
                old_df = pd.read_csv("../dat/tconst_scraped_data.csv")
            except FileNotFoundError:
                old_df = None

            if old_df is not None:
                new_df = new_df.append(old_df, ignore_index=True)

            new_df.to_csv("../dat/tconst_scraped_data.csv", index=False)

            self.buffer = []
