def visualize(sents, text, name):
    with open(r"data/" + name + ".txt", "w") as f:
        for k in range(len(sents)):
            final = []
            cnSent, enSent = sents[k]
            final.append(' '.join(cnSent) + '\n')
            final.append(' '.join(enSent) + '\n')

            for s in range(text[k].shape[0]):
                string = str(s) + ':'
                for i in range(len(enSent)):
                    string += cnSent[text[k][s, i]] + ' '
                final.append(string + '\n')

            f.writelines(final)
