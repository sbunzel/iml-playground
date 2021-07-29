FROM condaforge/mambaforge:4.9.2-7

ARG user_name
ARG user_id
ARG user_gid=${user_id}

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

# Create the user
RUN groupadd --gid $user_gid $user_name && \
    useradd --uid $user_id --gid $user_gid -ms /bin/bash $user_name && \
    chown -R ${user_id}:${user_gid} /opt/conda . && \
    # Add sudo support. Omit if you don't need to install software after connecting.
    apt-get update && \
    apt-get install -y sudo && \
    echo $user_name ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$user_name && \
    chmod 0440 /etc/sudoers.d/$user_name


USER $user_name

RUN sudo apt install -y \
    gcc \
    git \
    vim \
    ssh-client && \
    sudo rm -rf /var/lib/apt/lists/*

# Copy environment files to Image.
# Files on root level will overwrite other files that are within any folder.
COPY **/environment*.yml \
    environment*.yml \
    # Copy valid file to prevent failure of missing files
    */devcontainer.json \
    # target dir:
    ./

# Install dependencies
# Create environment files temporarily if not existent so that command not fails
RUN [[ -f environment-dev.yml ]] || echo "channels:" > environment-dev.yml && \
    [[ -f environment.yml ]] || echo "channels:" > environment.yml && \
    mamba env create -n devenv -f environment.yml && \
    mamba env update -n devenv -f environment-dev.yml && \
    mamba clean -afy && \
    # Remove copied or created files
    rm environment.yml environment-dev.yml devcontainer.json && \
    echo "conda activate devenv" >> ~/.bashrc

ENTRYPOINT ["conda", "run", "-n", "devenv", "/bin/bash", "-c"]
CMD ["while sleep 1000; do :; done"]
