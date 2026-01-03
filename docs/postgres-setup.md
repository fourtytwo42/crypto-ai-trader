## Postgres Setup

This project uses PostgreSQL via `DATABASE_URL`.

Local setup used for the VM:
- Role: `trading_user`
- Password: `password`
- Database: `bitcoin_trading`

SQL used to create role and database:
```
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='trading_user') THEN
    CREATE ROLE trading_user LOGIN PASSWORD 'password';
  END IF;
END
$$;

SELECT 'CREATE DATABASE bitcoin_trading OWNER trading_user'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'bitcoin_trading')\gexec

GRANT ALL PRIVILEGES ON DATABASE bitcoin_trading TO trading_user;
```

Reset role password (if needed):
```
ALTER ROLE trading_user PASSWORD 'password';
```

Default env:
- `DATABASE_URL=postgresql://trading_user:password@localhost:5432/bitcoin_trading`
