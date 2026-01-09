## Environment Variables

This repo uses environment variables for the pump.fun pipeline and the API/UI.

### Core

- `PUMPFUN_DATABASE_URL` (required)
  - Postgres connection string for pump.fun trades, tokens, and features.
  - Example: `postgresql://pumpfun_user:password@localhost:5432/pumpfun_db`

### Frontend

- `NEXT_PUBLIC_PUMPFUN_API_URL` (optional)
  - Base URL for the pump.fun API used by the Next.js frontend.
  - Default: `http://127.0.0.1:8081`

### API

The API reads `PUMPFUN_DATABASE_URL`. You can place it in `pumpfun_api/.env` or export it in your shell before running uvicorn.
