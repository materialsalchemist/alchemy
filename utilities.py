import logging

class Utilities:
    @staticmethod
    def _setup_logging() -> None:
        """
        Configure logging settings for the reaction network generator.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - ReactionNetwork::%(levelname)s - %(message)s",
        )
