# Privacy Notice

FutureSightML is a fully local application. Here's what that means for your data:

## What stays on your machine

- **All ML inference runs locally.** Your teams are evaluated by models running on your own hardware. No team data is sent to any external server.
- **Your database is local.** The SQLite database containing scraped replays lives in your `data/` directory and is never uploaded anywhere.
- **No accounts or tracking.** FutureSightML does not require sign-in, does not use analytics, and does not collect telemetry.

## What accesses the internet

- **Replay scraping** fetches publicly available battle replays from the [Pokemon Showdown replay API](https://replay.pokemonshowdown.com/). This is the same data anyone can view in a browser.
- **Pokemon sprites** are loaded from `play.pokemonshowdown.com` for display in the GUI. No user data is sent in these requests.
- **HuggingFace import** (optional) downloads a public dataset of replays. No user data is uploaded.

## Summary

No personal data, team compositions, or usage patterns ever leave your machine. FutureSightML is open-source — you can verify this yourself.
