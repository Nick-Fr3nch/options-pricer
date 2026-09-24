"""
Command-line interface for the options pricer.

Examples
--------
Price a European call:
    python cli.py price --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --type call

Show all Greeks:
    python cli.py greeks --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --type call

Compute implied vol from a market price:
    python cli.py iv --price 10.45 --S 100 --K 100 --T 1 --r 0.05 --type call

Price an American put with a binomial tree:
    python cli.py tree --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 \
        --type put --american --N 1000
"""

import argparse

from pricer.black_scholes import bs_price
from pricer.binomial import crr_price
from pricer.greeks import bs_greeks
from pricer.implied_vol import implied_vol


def _common(p):
    """Add the arguments shared by every subcommand."""
    p.add_argument("--S", type=float, required=True, help="Spot price")
    p.add_argument("--K", type=float, required=True, help="Strike")
    p.add_argument("--T", type=float, required=True, help="Time to expiry (years)")
    p.add_argument("--r", type=float, required=True, help="Risk-free rate (continuous)")
    p.add_argument("--q", type=float, default=0.0, help="Dividend yield")
    p.add_argument("--type", choices=["call", "put"], default="call")


def cmd_price(args):
    price = bs_price(args.S, args.K, args.T, args.r, args.sigma, args.q, args.type)
    print(f"{args.type.capitalize()} price: {price:.6f}")


def cmd_greeks(args):
    g = bs_greeks(args.S, args.K, args.T, args.r, args.sigma, args.q, args.type)
    print(f"Option type: {args.type}")
    print(f"Delta: {g['delta']:+.6f}")
    print(f"Gamma: {g['gamma']:.6f}")
    print(f"Vega : {g['vega']:.6f}   (per 1.00 vol; /100 for per 1%)")
    print(f"Theta: {g['theta']:.6f}  (per year;   /365 for per day)")
    print(f"Rho  : {g['rho']:+.6f}  (per 1.00 rate; /100 for per 1%)")


def cmd_iv(args):
    iv = implied_vol(args.price, args.S, args.K, args.T, args.r, args.q, args.type)
    print(f"Implied volatility: {iv:.6f}  ({iv * 100:.2f}%)")


def cmd_tree(args):
    price = crr_price(
        args.S, args.K, args.T, args.r, args.sigma, args.q,
        N=args.N, option=args.type, american=args.american,
    )
    label = "American" if args.american else "European"
    print(f"{label} {args.type} price (N={args.N}): {price:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Options pricer: Black-Scholes, Greeks, IV, binomial tree."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_price = sub.add_parser("price", help="Black-Scholes price")
    _common(p_price)
    p_price.add_argument("--sigma", type=float, required=True)
    p_price.set_defaults(func=cmd_price)

    p_greeks = sub.add_parser("greeks", help="Black-Scholes Greeks")
    _common(p_greeks)
    p_greeks.add_argument("--sigma", type=float, required=True)
    p_greeks.set_defaults(func=cmd_greeks)

    p_iv = sub.add_parser("iv", help="Implied volatility from market price")
    _common(p_iv)
    p_iv.add_argument("--price", type=float, required=True,
                      help="Observed market price of the option")
    p_iv.set_defaults(func=cmd_iv)

    p_tree = sub.add_parser("tree", help="Binomial tree price")
    _common(p_tree)
    p_tree.add_argument("--sigma", type=float, required=True)
    p_tree.add_argument("--N", type=int, default=500,
                        help="Number of time steps (higher = more accurate)")
    p_tree.add_argument("--american", action="store_true",
                        help="Enable early exercise (American option)")
    p_tree.set_defaults(func=cmd_tree)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()