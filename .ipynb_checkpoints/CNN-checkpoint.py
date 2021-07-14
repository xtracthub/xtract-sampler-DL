{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Jupyter Notebook Version of DL\n",
    "\n",
    "The project is first developed in Jupyter Notebook for easy testing/verification but could be moved to a formal Python Script in the future (if I have time). Contrary to the Xtract-Sampler we won't be implementing any byte extraction but rather right now assume we have the data.\n",
    "***\n",
    "Training and Developing a model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import Statements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Loading Files/Data Processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "loading files now...\n",
      "loading files done!\n"
     ]
    }
   ],
   "source": [
    "print(\"loading files now...\")\n",
    "\n",
    "with open('CDIACFileData/ByteVectors/byte_vector_dict_256KB_one_gram.pkl', \"rb\") as fp1:\n",
    "    one_gram = pickle.load(fp1)\n",
    "with open('CDIACFileData/ByteVectors/byte_vector_dict_256KB_two_gram.pkl', \"rb\") as fp2:\n",
    "    two_gram = pickle.load(fp2)\n",
    "\n",
    "print(\"loading files done!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
