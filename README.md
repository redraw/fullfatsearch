# yt-search-db

- crear cuenta en [Vercel](https://vercel.com)
- aca en Github, crear un nuevo repo desde este template (botón verde)
- agregar secrets:
  - parado en el repo, ir a Settings > Secrets and variables > Actions
  - agregar `VERCEL_TOKEN` (obtener en https://vercel.com/account/tokens)
- editar [config.yaml](./config.yaml)
- ver Actions

### datasette-auth-passwords
para habilitar user root con password:

- en [datasette.yaml](./datasette.yaml) descomentar la parte de `allow`
- en el proyecto de Vercel agregar una env `ROOT_PASSWORD_HASH`, para el valor:
  - generar una pass desde `https://<DEPLOY_URL>/-/password-tool`, la URL del deploy figura en algun run dentro Actions > último run > build > Deploy
