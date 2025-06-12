# formulas: multihead latent attention
```mark down
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
