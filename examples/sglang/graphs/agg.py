from components.frontend import Frontend
from components.processor import Processor
from components.worker import SGLangWorker

Frontend.link(Processor).link(SGLangWorker)