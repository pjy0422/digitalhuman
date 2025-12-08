from .dataloader import DataLoader
from . import dataloader_registry
import json, csv


@dataloader_registry.register("tasksolving/browsecomp/templates/")
class Browsecomp(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        num = 0
        with open(self.path, 'r') as f:
            for line in f:
                num += 1
                data = json.loads(line)
                self.examples.append(
                     {
                         'input': data['query'],
                         'answer': data['answer']
                     }
                )
                if num == 100:
                    break
        # with open(self.path, mode='r', newline='') as file:
        #         csv_reader = csv.reader(file)