from g2p_en import G2p
import numpy as np

puncts = ['.', ',', '?', '!']
stresses = ['0', '1', '2']

def clean_phonemes(phonemes):
    cleaned = [
        (p[:-1] if any([stress in p for stress in stresses]) else p)
        for p in phonemes
        if (p not in puncts)
    ]
    if cleaned[0] == ' ':
        cleaned = cleaned[1:]
    if cleaned[-1] == ' ':
        cleaned = cleaned[:-1]
    return cleaned

def levenshtein_distance(r, h):
    """
    taken from https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/utils/rnnEval.py
    """
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint16)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def mean_wer(decodedSentences, trueSentences):
    allWordErr = []
    allWord = []
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        trueWords = trueSent.split(" ")
        decWords = decSent.split(" ")
        nWordErr = levenshtein_distance(trueWords, decWords)

        allWordErr.append(nWordErr)
        allWord.append(len(trueWords))

    wer = np.sum(allWordErr) / np.sum(allWord)
    return wer, (allWordErr, allWord)

def mean_per(decodedSentences, trueSentences):
    allPhonErr = []
    allPhon = []
    transcriber = G2p()
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        truePhon = clean_phonemes(transcriber(trueSent))
        decPhon = clean_phonemes(transcriber(decSent))
        nPhonErr = levenshtein_distance(truePhon, decPhon)

        allPhonErr.append(nPhonErr)
        allPhon.append(len(truePhon))

    wer = np.sum(allPhonErr) / np.sum(allPhon)
    return wer, (allPhonErr, allPhon)

def sentences_to_phonemes(sentences):
    transcriber = G2p()
    phonemes = []
    for sentence in sentences:
        phoneme_list = clean_phonemes(transcriber(sentence))
        sentence_phonemes = ""
        last_phoneme = False
        for phoneme in phoneme_list:
            if phoneme == " ":
                sentence_phonemes += phoneme
                last_phoneme = False
            else:
                sentence_phonemes += ("-" if last_phoneme else "") + phoneme
                last_phoneme = True
        phonemes.append(sentence_phonemes)
    return phonemes