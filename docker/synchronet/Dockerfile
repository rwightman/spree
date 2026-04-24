ARG SYNCHRONET_IMAGE=bbsio/synchronet:3.19c
FROM ${SYNCHRONET_IMAGE}

USER root

RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            ca-certificates \
            dos2unix \
            procps \
            unzip \
            p7zip-full; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends dosemu2 \
            || DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends dosemu; \
        rm -rf /var/lib/apt/lists/*; \
    elif command -v apk >/dev/null 2>&1; then \
        apk add --no-cache ca-certificates dos2unix dosemu procps p7zip unzip; \
    else \
        echo "No supported package manager found for DOSEMU packages" >&2; \
        exit 1; \
    fi

COPY docker/synchronet/dosemu.conf /etc/dosemu/dosemu.conf

