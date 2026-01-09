## Frontend Setup

The pump.fun UI lives in `pumpfun_frontend/` and talks to the FastAPI service.

### Install + run

```bash
cd pumpfun_frontend
npm install
npm run dev
```

The app runs on `http://127.0.0.1:3001` by default.

### API configuration

Set `NEXT_PUBLIC_PUMPFUN_API_URL` if the API is not on `http://127.0.0.1:8081`.
