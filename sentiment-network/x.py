import numpy as np

def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

print(len(reviews))

print(reviews[0])


print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)

from collections import Counter

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for label,line in zip(labels,reviews):
    words = line.split(' ')
    for word in line.split(' '):
        if label == "POSITIVE":
            positive_counts[word] = positive_counts[word] + 1
        elif label == "NEGATIVE":
            negative_counts[word] = negative_counts[word] + 1
        else:
            print("ERROR")
        total_counts[word] = total_counts[word] + 1


print(positive_counts.most_common(5))
print(negative_counts.most_common(5))
print(total_counts.most_common(5))

# Create Counter object to store positive/negative ratios
pos_neg_ratios = Counter()

# TODO: Calculate the ratios of positive and negative uses of the most common words
#       Consider words to be "common" if they've been used at least 100 times

for word in positive_counts.keys():
    a = positive_counts[word]
    b = negative_counts[word]
    pos_neg_ratios[word] = a / (b + 1)

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

# TODO: Convert ratios to logs
for word in pos_neg_ratios.keys():
    pos_neg_ratios[word] = np.log(pos_neg_ratios[word])

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

print(pos_neg_ratios.most_common(5))
print(list(reversed(pos_neg_ratios.most_common()))[0:5])

