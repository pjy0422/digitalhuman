from .dataloader import DataLoader
from . import dataloader_registry
import json, csv


@dataloader_registry.register("tasksolving/brainstorming/templates/1v1/")
class Debate_persuasion(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        with open(self.path, mode='r', newline='') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)  # Read the header row
                # print(f"Header: {header}")
                for row in csv_reader:
                    di = {
                        'input': "compose a brief argumentative essay on: "+row[1],
                        'answer': ''
                    }
                    self.examples.append(di)
