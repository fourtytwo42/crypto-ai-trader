## Postgres Setup

This branch uses PostgreSQL for pump.fun trades and model inputs via `PUMPFUN_DATABASE_URL`.

Example local setup:
- Role: `pumpfun_user`
- Password: `password`
- Database: `pumpfun_db`

SQL used to create role and database:
```
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='pumpfun_user') THEN
    CREATE ROLE pumpfun_user LOGIN PASSWORD 'password';
  END IF;
END
$$;

SELECT 'CREATE DATABASE pumpfun_db OWNER pumpfun_user'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'pumpfun_db')\gexec

GRANT ALL PRIVILEGES ON DATABASE pumpfun_db TO pumpfun_user;
```

Reset role password (if needed):
```
ALTER ROLE pumpfun_user PASSWORD 'password';
```

Default env:
- `PUMPFUN_DATABASE_URL=postgresql://pumpfun_user:password@localhost:5432/pumpfun_db`
