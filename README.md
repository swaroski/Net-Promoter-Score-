# Net-Promoter-Score
NPS stands for Net Promoter Score. It's a customer satisfaction benchmark that measures how likely your customers are to recommend your business to a friend. 
It is a typical benchmark companies measure to evaluate and improve customer loyalty. 
The survey asks its customers, "On a scale of 0 to 10, how likely are you to recommend us to someone". 
Scores 0-6 are Detractors, scores 7-8 are Passives, and scores 9-10 are Promoters.
Disregarding the Passives, subtract the percentage of Detractor responses from the percentage of Promoter responses to determine your Net Promoter Score. This score can range from -100 to 100.

Here, we have used LSTM(Long Short Term Memory) and GRU(Gated Recurrent Units). These forms a better way of utilizing RNNs to perform sentiment analysis on text. Since, we are trying to derive sentiment from product reviews, our neural network would do well to remember only important aspects of the reviews and bring them forth to form an opinion of the snetiment behind such reviews. The networks also helps us address the vanishing gradient problem during back propogation. They have an internal mechanism called gates which decide which information to keep and which to throw away. By doing that, it learns to use relevant information to make predictions.  
