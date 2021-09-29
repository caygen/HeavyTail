"""
Copyright c 2021 by Northwestern University. All Rights Reserved.

@author: Can Aygen
"""
class HT_Fun:
    def __init__(self): #set up the dependencies

        self.members = ['Tiger', 'Elephant', 'Wild Cat']


    def printMembers(self): #test of classes
        print('Printing members of the Mammals class')
        for member in self.members:
            print('\t%s ' % member)
