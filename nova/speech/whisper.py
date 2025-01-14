from typing import Literal

import numpy as np
from pydantic import BaseModel, PrivateAttr, validate_call

import pyaudio
import whisper
import torch

from nova.utils import set_device


class TranscribeParameters(BaseModel):
    """
    Config parameters for `WhisperLiveTranscriber`.

    """

    silence_limit: float = 2.0
    silence_threshold: float = 5.0
    chunk_size: int = 1024
    sample_rate: int = 16000
    channels: int = 1

    _chunks_per_process: int | None = PrivateAttr(default=None)

    @property
    def chunks_per_process(self) -> int:
        if self._chunks_per_process is not None:
            return self._chunks_per_process

        chunks = int(self.sample_rate / self.chunk_size * self.silence_limit)
        self._chunks_per_process = chunks
        return chunks


class WhisperLiveTranscriber:
    """
    Performs live transcription using the OpenAI Whisper model.

    Args:
        model_size: (Literal["tiny", "base"]): the size of the model to use. Limited to small scale models
        params (TranscribeParameters, optional): config parameters for the transcriber. Optimal parameters set by default
        device (str, optional): the type of `torch.device` to use. Default is 'auto'
    """

    @validate_call
    def __init__(
        self,
        model_size: Literal["tiny", "base"],
        *,
        params: TranscribeParameters = TranscribeParameters(),
        device: str = "auto",
    ) -> None:
        self.device = set_device(device)
        self.params = params

        self.model = whisper.load_model(model_size).to(self.device)

        self.audio = None
        self.stream = None
        self.frames = []
        self.silent_chunks = 0
        self.is_recording = False

    def _init_audio(self) -> None:
        """Initialize PyAudio and open a stream."""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.params.channels,
            rate=self.params.sample_rate,
            input=True,
            frames_per_buffer=self.params.chunks_per_process,
        )

    def _cleanup(self) -> None:
        """Clean up audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.audio:
            self.audio.terminate()

    def _is_silent(self, audio_data: np.ndarray) -> bool:
        """Check if the audio chunk is below the volume threshold."""
        volume_norm = np.linalg.norm(audio_data) / len(audio_data)
        is_silent = volume_norm < self.params.silence_threshold

        print(
            f"\rVolume: {volume_norm:.2f} {'[Silent]' if is_silent else '[Active]'}",
            end="",
        )

        return is_silent

    def _process_audio(self, frames: list) -> str | None:
        """Process collected audio frames and return the transcription."""
        if not frames:
            return None

        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

        # Normalize [-1, 1]
        audio_data = audio_data.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_data).to(self.device)

        result = self.model.transcribe(audio_tensor)
        return result["text"]

    def _read_audio_chunk(self) -> tuple[bytes, np.ndarray]:
        """Read a chunk of audio data and return both raw and numpy format."""
        data = self.stream.read(self.params.chunk_size, exception_on_overflow=False)
        return data, np.frombuffer(data, dtype=np.int16)

    def transcribe(self) -> None:
        """Start live transcription."""
        try:
            # Initialize audio system
            self._init_audio()
            self.stream.start_stream()
            print(
                f"Listening... Will transcribe after {self.params.silence_limit} seconds of silence. Press Ctrl+C to stop."
            )
            print(f"Volume threshold: {self.params.silence_threshold}")

            while True:
                # Read and process audio chunk
                raw_data, audio_chunk = self._read_audio_chunk()

                # Update silence detection
                if self._is_silent(audio_chunk):
                    self.silent_chunks += 1

                    # If we've been recording and hit silence limit, process audio
                    if self.silent_chunks >= self.params.chunks_per_process:
                        if self.is_recording:
                            print("\nProcessing...")

                            # Process audio and get transcription
                            transcription = self._process_audio(self.frames)

                            if transcription:
                                print("Transcription:", transcription)

                            # Reset recording state
                            self.frames = []
                            self.is_recording = False
                else:
                    self.silent_chunks = 0
                    self.is_recording = True

                # If we're recording, store the audio
                if self.is_recording:
                    self.frames.append(raw_data)

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self._cleanup()


def main() -> None:
    model = WhisperLiveTranscriber(model_size="tiny")
    model.transcribe()


if __name__ == "__main__":
    main()
