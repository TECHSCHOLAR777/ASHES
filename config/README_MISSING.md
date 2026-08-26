# Missing configuration

These values were not provided at build time. The system runs without them in a degraded
mode described below. Fill in `.env` and restart when available.

## FIRMS_MAP_KEY

Not provided. Register a free key at
https://firms.modaps.eosdis.nasa.gov/api/map_key/ and set `FIRMS_MAP_KEY` in `.env`.

Default behavior without it: `src/clients/firms.py` raises `FIRMSKeyMissing` internally,
the caller catches it, sets the `firms_unavailable` flag on the ActionCard, and the policy
engine treats FIRMS counts as zero/unknown rather than fabricating hotspot data.

## SLACK_BOT_TOKEN

Not provided. Default behavior: `src/delivery/slack_delivery.py` runs in log-only mode,
it writes the message payload to the daily JSONL log instead of calling the Slack API,
and returns a synthetic `thread_ts` derived from the card_id so the ResponseCard threading
logic still works end to end.

## SMTP_HOST / SMTP_USER / SMTP_PASS / SMTP_FROM / SMTP_TO

Not provided. Default behavior: `src/delivery/email_delivery.py` runs in log-only mode,
writing the composed email body to the daily JSONL log instead of calling `smtplib`.
