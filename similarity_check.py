from difflib import SequenceMatcher
 
# Utility function to compute similarity
def similar(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()
 
# Initializing strings
test_string1 = 'PM of India'
test_string2 = 'CM of India'
 
# using SequenceMatcher.ratio()
# similarity between strings
res = similar(test_string1, test_string2)


# printing the result
print ("The similarity between 2 strings is : " + str(res))