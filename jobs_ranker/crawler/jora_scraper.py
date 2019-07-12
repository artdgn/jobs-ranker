import scrapy


class JoraSpider(scrapy.Spider):
    name = 'jora-spider'
    base_url = 'https://au.jora.com'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = kwargs.get('start_urls').split(',')

    def parse(self, response):
        for job in response.css('#jobresults').css('.jwrap'):
            rel_url = job.css(".jobtitle").css('a::attr(href)').extract_first()
            rel_url = rel_url.split('?')[0] if rel_url else None
            full_url = (self.base_url + rel_url) if rel_url else None
            item = {
                'title': job.css(".jobtitle::text").extract_first(),
                'url': full_url,
                'salary': job.css(".salary::text").extract_first(),
                'date': job.css(".date::text").extract_first(),
                'company': job.css(".company::text").extract_first(),
            }
            if full_url:
                yield scrapy.Request(full_url,
                                     callback=self.parse_job_page,
                                     meta=item)
            else:
                yield item

        next_page_url = response.css('.next_page').css('a::attr(href)').extract_first()
        if next_page_url is not None:
            yield scrapy.Request(response.urljoin(self.base_url + next_page_url))

    @staticmethod
    def parse_job_page(response):
        item = response.meta
        item['description'] = '\n'.join(response.css('.summary ::text').extract())
        yield item

