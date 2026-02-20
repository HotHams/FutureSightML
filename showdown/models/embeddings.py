"""Learned embedding layers for Pokemon, moves, items, abilities."""

import torch
import torch.nn as nn


class PokemonEncoder(nn.Module):
    """Encode a single Pokemon set into a dense representation.

    Input per Pokemon:
        - species index (int)
        - ability index (int)
        - item index (int)
        - 4 move indices (int)

    Output: dense vector of size `output_dim`.
    """

    def __init__(
        self,
        num_species: int,
        num_moves: int,
        num_items: int,
        num_abilities: int,
        species_dim: int = 64,
        move_dim: int = 32,
        item_dim: int = 32,
        ability_dim: int = 32,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.species_embed = nn.Embedding(num_species, species_dim, padding_idx=0)
        self.move_embed = nn.Embedding(num_moves, move_dim, padding_idx=0)
        self.item_embed = nn.Embedding(num_items, item_dim, padding_idx=0)
        self.ability_embed = nn.Embedding(num_abilities, ability_dim, padding_idx=0)

        input_dim = species_dim + 4 * move_dim + item_dim + ability_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        species: torch.Tensor,     # (batch, 6)
        moves: torch.Tensor,       # (batch, 6, 4)
        items: torch.Tensor,       # (batch, 6)
        abilities: torch.Tensor,   # (batch, 6)
    ) -> torch.Tensor:
        """Encode all 6 Pokemon on a team.

        Returns: (batch, 6, output_dim)
        """
        batch_size = species.size(0)
        team_size = species.size(1)

        sp_emb = self.species_embed(species)          # (B, 6, sp_dim)
        it_emb = self.item_embed(items)               # (B, 6, it_dim)
        ab_emb = self.ability_embed(abilities)        # (B, 6, ab_dim)

        # Moves: (B, 6, 4) -> (B, 6, 4, mv_dim) -> concat to (B, 6, 4*mv_dim)
        mv_emb = self.move_embed(moves)               # (B, 6, 4, mv_dim)
        mv_emb = mv_emb.view(batch_size, team_size, -1)  # (B, 6, 4*mv_dim)

        # Concatenate all embeddings per Pokemon
        x = torch.cat([sp_emb, mv_emb, it_emb, ab_emb], dim=-1)  # (B, 6, input_dim)

        # Apply MLP to each Pokemon independently
        x = self.mlp(x)  # (B, 6, output_dim)
        return x


class TeamEncoder(nn.Module):
    """Encode a team of 6 Pokemon into a single team representation using self-attention.

    Takes the output of PokemonEncoder (batch, 6, pokemon_dim) and produces
    a single team vector (batch, team_dim) that captures team-level synergy.
    """

    def __init__(
        self,
        pokemon_dim: int = 128,
        team_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pokemon_dim,
            nhead=num_heads,
            dim_feedforward=pokemon_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Attention pooling: learn to weight the 6 Pokemon
        self.attn_pool = nn.Sequential(
            nn.Linear(pokemon_dim, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(pokemon_dim, team_dim),
            nn.LayerNorm(team_dim),
            nn.GELU(),
        )

    def forward(
        self,
        pokemon_reprs: torch.Tensor,  # (batch, 6, pokemon_dim)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce team representation.

        Args:
            pokemon_reprs: Output of PokemonEncoder, shape (B, 6, D)
            mask: Optional bool mask of shape (B, 6), True = padded/absent

        Returns: (batch, team_dim)
        """
        # Self-attention over the 6 Pokemon
        x = self.transformer(pokemon_reprs, src_key_padding_mask=mask)  # (B, 6, D)

        # Attention pooling
        attn_weights = self.attn_pool(x)  # (B, 6, 1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=1)  # (B, 6, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, D)

        return self.projection(pooled)  # (B, team_dim)
