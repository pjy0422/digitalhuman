from .dataloader import DataLoader
from . import dataloader_registry
import json, csv


@dataloader_registry.register("tasksolving/research_questions/templates/")
class Research_questions(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        with open(self.path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.examples.append(
                     {
                         'input': data['question'],
                         'answer': ''
                     }
                )
        # with open(self.path, mode='r', newline='') as file:
        #         csv_reader = csv.reader(file)
        #         header = next(csv_reader)  # Read the header row
        #         # print(f"Header: {header}")
        #         for row in csv_reader:
        #             di = {
        #                 'input': row[1],
        #                 'answer': ''
        #             }
        #             self.examples.append(di)
