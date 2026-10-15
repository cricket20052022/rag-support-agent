# API and Rate Limits
API keys are created in Settings > Developers and can be scoped to read-only or read-write access.
The Starter plan allows 60 API requests per minute, Pro allows 600 and Business allows 3000.
When the rate limit is exceeded the API returns HTTP 429 with a Retry-After header.
Webhooks are retried up to 5 times with exponential backoff.
The API base URL is https://api.northwind.example/v2.
