# FROM node:14 AS builder
# WORKDIR /code
# COPY . ./code
# RUN npm install && npm build
# VOLUME .:./code


# FROM nginx:alpine
# WORKDIR /usr/share/nginx/html
# RUN rm -rf ./*
# COPY --from=builder /code/dist .
# ENTRYPOINT ["nginx", "-g", "daemon off;"]

# EXPOSE 8083

FROM node:14 AS builder
WORKDIR /app
COPY . .
# RUN yarn install && yarn build
RUN npm install && npm run build





FROM nginx:alpine
WORKDIR /usr/share/nginx/html
RUN rm -rf ./*
COPY --from=builder /app/dist .
ENTRYPOINT ["nginx", "-g", "daemon off;"]

EXPOSE 80
