# syntax=docker/dockerfile:1

ARG UBUNTU_IMAGE=ubuntu:24.04

FROM ${UBUNTU_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG SYNCHRONET_REPOSITORY=https://github.com/SynchronetBBS/sbbs.git
ARG SYNCHRONET_COMMIT=74911bb1379078e52a52f4cac47696ca4190d3f9

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        libarchive-dev \
        libcap2-bin \
        libmosquitto-dev \
        libncurses-dev \
        libnspr4-dev \
        libsqlite3-dev \
        libssl-dev \
        pkgconf \
        python3 \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sbbs/repo
RUN git init \
    && git remote add origin "${SYNCHRONET_REPOSITORY}" \
    && git fetch --depth 1 origin "${SYNCHRONET_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${SYNCHRONET_COMMIT}"

COPY docker/synchronet/build-synchronet.sh /usr/local/sbin/build-synchronet
RUN chmod 0755 /usr/local/sbin/build-synchronet \
    && SYNCHRONET_COMMIT="${SYNCHRONET_COMMIT}" /usr/local/sbin/build-synchronet

FROM ${UBUNTU_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG DOSEMU2_VERSION=2.0~pre9-10228-0aeb4d174+202607270023~ubuntu24.04.1

ENV TERM=xterm \
    SBBSCTRL=/sbbs/ctrl \
    SBBSEXEC=/sbbs/exec \
    LD_LIBRARY_PATH=/sbbs/exec \
    PATH=/sbbs/scripts:/sbbs/exec:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        dos2unix \
        gnupg \
        less \
        libarchive-tools \
        libarchive13t64 \
        libcap2-bin \
        libffi8 \
        libmosquitto1 \
        libncursesw6 \
        libnspr4 \
        libsqlite3-0 \
        libssl3t64 \
        p7zip-full \
        procps \
        sudo \
        unzip \
        util-linux \
        zip \
    && rm -rf /var/lib/apt/lists/*

COPY docker/synchronet/install-dosemu2.sh /usr/local/sbin/install-dosemu2
RUN chmod 0755 /usr/local/sbin/install-dosemu2 \
    && DOSEMU2_VERSION="${DOSEMU2_VERSION}" /usr/local/sbin/install-dosemu2 \
    && ln -s /usr/bin/dosemu /usr/bin/dosemu.bin

RUN groupmod --new-name sbbs ubuntu \
    && usermod --login sbbs --home /home/sbbs --move-home --gid sbbs ubuntu \
    && usermod --append --groups sudo sbbs \
    && printf '%%sudo ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/sbbs \
    && chmod 0440 /etc/sudoers.d/sbbs

COPY --from=build /sbbs /sbbs
COPY docker/synchronet/scripts/ /sbbs/scripts/
COPY docker/synchronet/dosemu.conf /sbbs/dist/ctrl/dosemu.conf

RUN chmod 0755 /sbbs/scripts/* \
    && chown -R sbbs:sbbs /sbbs \
    && setcap cap_net_bind_service=+ep /sbbs/exec/sbbs \
    && ln -s /sbbs-data/backup /backup

WORKDIR /sbbs
USER sbbs

VOLUME ["/sbbs-data"]
CMD ["/sbbs/scripts/sbbs"]
