# formulas: multihead latent attention

for attention deepseek v3 adiots the MLA architecture. Let d donate the embedding dimension, nh denote the number of attention heads, dh denote the dimension per head, and ht ∈ Rd denote the attention input for the t-th token at a given attention layer.  The core of MLA is the low-rank joint compression for attention keys and values to reduce Key-Value (KV) cache during inference.

```mark down
W = matrix
h = vector
c = smaller vector
k = key value
                       k   dkv
                      c  = W  h ,               (1)
                       t       t

   c     c         c        uk kv
 [k   ; k   ;...; k    ] = W  c  ,              (2)
   t,1   t,2       t,n         t
                      h

                      i r        KR
                       k = RoPE(w  h ),         (3)
                        t            t

                             c     R
                     k   = [k   ; k ],          (4)
                      t,i    t,i   t

  c     c          c        c   uv kv
[v   ; v   ;....; v    ] = v = w  c  ,          (5)
  t,1   t,2        t,n      t      t
                      h

                            7
```


```mark down
                        Token embedding vector,
                        for example "Dog"
h =[423,634,634,...234]
 t
              1 x 7168

          |542 819 138 ... 624|
    DKV   |983 117 756 ... 853|
@  W   =  |467 982 751 ... 394|
          |. . . . . . . . . .|
          |219 756 627 ... 853|
                    7168 x 576

          {Reduced Vector}
=> C = [534, 623, 846,... 624]
    t                  1 x 576
```
