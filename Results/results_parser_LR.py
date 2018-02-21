# @author Ben Greenawald

# File to parse the results of capstone tests


# Create the violent/non-violent group labels
group_labels_dir = "C:/Users/bgree/Documents/capstone/Eng/eng_group_labels.txt"
group_labels = {}

with open(group_labels_dir, "r") as file:
    for line in file.readlines():
        group_label = line.split(",")
        group_labels[group_label[0].strip()] = int(group_label[1])
    file.close()

print(group_labels)

# Parse the actual results
results_file ="C:/Users/bgree/Documents/capstone/Results/Logistic Regression/2-15-18.txt"

# Lists to hold f-1 scores
pos_f1 = []
neg_f1 = []

with open(results_file, "r") as file:

    lines = file.readlines()
    for line in range(0, len(lines)):

        # Process in groups of 4
        if (line + 2) % 4 == 0:
            group = lines[line].split(",")[0].split(" ")[1].strip()
            if group_labels[group] == 0:
   #             res_line = lines[line+2].split(": ")
    #            res = res_line[1].strip()
                temp = lines[line+1].split("=")
                neg_f1.append(float(temp[3]))
            else:
     #           res_line = lines[line+2].split(": ")
      #          res = res_line[1].strip()
                temp = lines[line+2].split("=")
                pos_f1.append(float(temp[3]))
    file.close()

print("Violent group average F-1 " + str(sum(pos_f1)/len(pos_f1)))
print("Non-Violent group average F-1: " + str(sum(neg_f1)/len(neg_f1)))