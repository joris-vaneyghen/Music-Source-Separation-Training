# Use the official Python image as a base
FROM python:3.12

# Create non-root user
RUN useradd -m -u 1000 user

# Switch to non-root user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/workdir

COPY --chown=user ./requirements.txt requirements.txt

# Install requirements except pyaudio and wxpython as we don't need the GUI
RUN sed -i '/pyaudio/d;/wxpython/d' requirements.txt && \
    pip install --no-cache-dir --upgrade -r requirements.txt && \
    pip install --no-cache-dir --upgrade kaggle

RUN mkdir checkpoints
RUN mkdir dataset

# Copy rest of the code
COPY --chown=user . $HOME/workdir

CMD [ "bash" ]