# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from attr import dataclass


class RetrievalMethod:
    @dataclass
    class Mode:
        TRAINING = 0
        LEARNING = 1
        TESTING = 2
    
    def add_image_pool(self):
        pass
    
    def learn_concepts(self):
        pass
    
    def retrieve(self):
        pass
    
    def set_mode(self, mode:Mode):
        self.mode = mode