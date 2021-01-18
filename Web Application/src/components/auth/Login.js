import React, { useState, useEffect, useContext } from "react";

import { Container } from "react-bootstrap";
import { useHistory } from "react-router-dom";
import UserContext from "../../context/UserContext";
import Axios from "axios";
import ErrorNotice from "../misc/ErrorNotice";

import "./auth.css";

export default function Login() {
	const [email, setEmail] = useState();
	const [password, setPassword] = useState();
	const [isAdmin, setIsAdmin] = useState(false);
	const [error, setError] = useState();

	// const [authorized, setAuthorized] = useState(false);

	const { userData, setUserData } = useContext(UserContext);
	const history = useHistory();

	useEffect(() => {
		console.log(userData.user === true);
		if (userData.user) history.push("/");
	}, [history, userData.user]);

	const submit = async (e) => {
		e.preventDefault();
		let serverUrl = "http://localhost:10000/users/login";
		if (isAdmin) serverUrl = "http://localhost:10000/users/login/admin";
		try {
			const loginUser = { email, password };
			console.log(loginUser);
			const loginRes = await Axios.post(serverUrl, loginUser);
			setUserData({
				token: loginRes.data.token,
				user: loginRes.data.user,
			});
			localStorage.setItem("auth-token", loginRes.data.token);
			history.push("/");
			// setAuthorized(true);
		} catch (err) {
			err.response.data.msg && setError(err.response.data.msg);
			console.log(err);
		}
	};
	// if (authorized) return <Redirect to="/webcam" />;
	return (
		<Container>
			<h2>Log in</h2>
			{error && <ErrorNotice message={error} clearError={() => setError(undefined)} />}
			<form className="form" onSubmit={submit}>
				<label htmlFor="login-email">Email</label>
				<input id="login-email" type="email" onChange={(e) => setEmail(e.target.value)} />

				<label htmlFor="login-password">Password</label>
				<input id="login-password" type="password" onChange={(e) => setPassword(e.target.value)} />

				<label htmlFor="admin"> Login as Admin</label>
				<input
					type="checkbox"
					id="admin"
					name="admin"
					value="admin"
					onChange={(e) => {
						setIsAdmin(e.target.checked);
					}}
				/>

				<input type="submit" value="Log in" />
			</form>
		</Container>
	);
}
