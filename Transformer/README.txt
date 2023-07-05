Reproduced the paper code：Attention is all you need

tutorial：https://www.youtube.com/watch?v=U0s0f995w14

Attention score: attention = softmax(QK^T/(d**0.5))V


Position Embedding: PE(pos,2i) = sin(pos/10000^(2i/d_model))  ;  PE(pos,2i+1) = cos(pos/10000^(2i/d_model)) . 
		[In the tutorial, we only do simple positional coding such as 0, 1, 2, and 3...]
