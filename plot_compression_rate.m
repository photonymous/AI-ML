% This file plots the compression rate achieved when tokenizing 
% a subset of Gutenberg English books into different vocabulary sizes.
% The tokenizer Hugging Face's BPE (ByteLevel) tokenizer, trained on
% roughly 18GB of Gutenberg English books.

x=[256,512,1024,2048,4096];
y=21497466./[12316832, 19825009/2, 17252937/2,15323093/2,13876501/2];
plot(x,y,'o');
hold on;
plot(x,y);
hold off;
grid on; 
xlabel('Vocab Size');
ylabel('Compression Rate');
title('Compression Rate vs Vocab Size on 21MB of Gutenberg English Books');


